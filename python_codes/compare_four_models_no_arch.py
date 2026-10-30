#!/usr/bin/env python3
"""Four-model Reliance Monte Carlo comparison without the third-party `arch` package.

Models:
  1. Normal IID
  2. Power-law hybrid IID
  3. GARCH(1,1) with Normal innovations
  4. GARCH(1,1) with power-law-hybrid standardized innovations

Uses NumPy, pandas, SciPy and Matplotlib. Parameters are fitted on training data only.
The existing test period is used for evaluation, not fitting.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

# ----------------------------- CONFIG ---------------------------------
PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
TRAIN_FILE = Path.home() / "reliance_train.csv"
TEST_FILE = Path.home() / "reliance_test.csv"
OUT_DIR = PROJECT_DIR / "four_model_results"
N_PATHS = 10_000
SEED = 42
RAW_TAIL_THRESHOLD = 0.025        # 2.5% raw daily log-return threshold (provisional)
STD_TAIL_THRESHOLD = 2.0          # standardized residual units (provisional)
# ----------------------------------------------------------------------


def find_column(df, candidates):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        name = str(c).strip().lower().replace(" ", "_")
        if any(x.strip().lower().replace(" ", "_") in name for x in candidates):
            return c
    return None


def load_returns(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Update TRAIN_FILE/TEST_FILE at the top of this script."
        )
    df = pd.read_csv(path)
    date_col = find_column(df, ["date", "datetime", "timestamp"])
    ret_col = find_column(df, ["log_return", "log_returns", "return", "returns", "daily_return"])
    close_col = find_column(df, ["close", "adj_close", "adj close", "price"])
    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col is not None else None

    if ret_col is not None:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        vals = r.dropna().to_numpy()
        if len(vals) and np.nanpercentile(np.abs(vals), 95) > 0.5:
            warnings.warn(f"{path.name}: returns appear to be in percent units; converting to decimals.")
            r = r / 100.0
        out = pd.DataFrame({"return": r})
        if dates is not None:
            out["date"] = dates
        if close_col is not None:
            out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        return out.dropna(subset=["return"]).reset_index(drop=True)

    if close_col is not None:
        close = pd.to_numeric(df[close_col], errors="coerce")
        out = pd.DataFrame({"return": np.log(close / close.shift(1)), "close": close})
        if dates is not None:
            out["date"] = dates
        return out.dropna(subset=["return"]).reset_index(drop=True)

    raise ValueError(f"No return or close-price column found in {path.name}: {list(df.columns)}")


def pareto_alpha(abs_tail, threshold):
    x = np.asarray(abs_tail, dtype=float)
    x = x[np.isfinite(x) & (x >= threshold)]
    if len(x) < 30:
        raise ValueError(f"Too few tail observations at threshold {threshold}.")
    denom = np.sum(np.log(x / threshold))
    if denom <= 0:
        raise ValueError("Invalid Pareto tail sample.")
    return float(len(x) / denom)


def fit_hybrid(r, threshold):
    r = np.asarray(r, dtype=float)
    mask = np.abs(r) >= threshold
    if int(mask.sum()) < 30:
        raise ValueError(f"Too few observations beyond threshold {threshold}.")
    return {
        "threshold": float(threshold),
        "alpha": pareto_alpha(np.abs(r[mask]), threshold),
        "p_tail": float(mask.mean()),
        "tail_sign_negative": float(np.mean(r[mask] < 0)),
        "body": r[~mask],
        "tail_n": int(mask.sum()),
    }


def draw_hybrid(params, shape, rng):
    n = int(np.prod(shape))
    tail = rng.random(n) < params["p_tail"]
    out = np.empty(n, dtype=float)
    body_n = int((~tail).sum())
    tail_n = int(tail.sum())
    if body_n:
        out[~tail] = rng.choice(params["body"], size=body_n, replace=True)
    if tail_n:
        u = np.clip(rng.random(tail_n), 0.0, 1.0 - 1e-12)
        mag = params["threshold"] * (1.0 - u) ** (-1.0 / params["alpha"])
        neg = rng.random(tail_n) < params["tail_sign_negative"]
        out[tail] = np.where(neg, -mag, mag)
    return out.reshape(shape)


def fit_garch11_normal(train_returns):
    """Gaussian QMLE for GARCH(1,1), using returns scaled to percent."""
    y = np.asarray(train_returns, dtype=float) * 100.0
    mu0 = float(np.mean(y))
    resid0 = y - mu0
    v0 = max(float(np.var(resid0)), 1e-6)
    # Optimize unconstrained transformed parameters:
    # omega=exp(a), alpha=sigmoid(b), beta=sigmoid(c)*(1-alpha)*0.999
    def unpack(theta):
        omega = np.exp(np.clip(theta[0], -20, 10))
        alpha = 1.0 / (1.0 + np.exp(-np.clip(theta[1], -20, 20)))
        beta_raw = 1.0 / (1.0 + np.exp(-np.clip(theta[2], -20, 20)))
        beta = beta_raw * max(1e-8, 0.999 - alpha)
        mu = theta[3]
        return mu, omega, alpha, beta

    def likelihood(theta):
        mu, omega, alpha, beta = unpack(theta)
        eps = y - mu
        h = np.empty_like(y)
        h[0] = max(v0, omega / max(1e-5, 1 - alpha - beta))
        for t in range(1, len(y)):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        h = np.maximum(h, 1e-10)
        return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + eps**2 / h))

    # Typical starting values for daily percent returns.
    start = np.array([np.log(max(v0 * 0.05, 1e-5)), -1.5, 2.0, mu0])
    result = minimize(likelihood, start, method="Nelder-Mead",
                      options={"maxiter": 12000, "xatol": 1e-7, "fatol": 1e-7})
    if not result.success:
        # Still accept a finite, sensible quasi-likelihood solution with a warning.
        warnings.warn(f"GARCH optimizer note: {result.message}")
    mu, omega, alpha, beta = unpack(result.x)
    eps = y - mu
    h = np.empty_like(y)
    h[0] = max(v0, omega / max(1e-5, 1 - alpha - beta))
    for t in range(1, len(y)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    h = np.maximum(h, 1e-10)
    z = eps / np.sqrt(h)
    z = (z - z.mean()) / z.std(ddof=1)
    return {
        "mu": float(mu), "omega": float(omega), "alpha": float(alpha),
        "beta": float(beta), "last_eps": float(eps[-1]), "last_var": float(h[-1]),
        "std_resid": z, "optimizer_success": bool(result.success),
        "objective": float(result.fun),
    }


def draw_standardized_hybrid(z, threshold):
    z = np.asarray(z, dtype=float)
    return fit_hybrid(z, threshold)


def garch_paths(params, n_paths, horizon, rng, innovation="normal", innovation_params=None):
    mu = params["mu"]
    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
    eps_prev = np.full(n_paths, params["last_eps"], dtype=float)
    var_prev = np.full(n_paths, params["last_var"], dtype=float)
    out = np.empty((n_paths, horizon), dtype=np.float32)
    for t in range(horizon):
        var_t = np.maximum(omega + alpha * eps_prev**2 + beta * var_prev, 1e-12)
        if innovation == "normal":
            z = rng.standard_normal(n_paths)
        else:
            z = draw_hybrid(innovation_params, (n_paths,), rng)
            # Rescale the empirical-body/Pareto-tail mixture to approximately unit variance.
            z = (z - innovation_params["norm_mean"]) / innovation_params["norm_sd"]
        eps_t = np.sqrt(var_t) * z
        out[:, t] = (mu + eps_t) / 100.0
        eps_prev, var_prev = eps_t, var_t
    return out


def path_metrics(returns, start_price):
    final_prices = start_price * np.exp(np.sum(returns, axis=1))
    running = start_price * np.exp(np.cumsum(returns, axis=1))
    with_start = np.column_stack([np.full(len(returns), start_price), running])
    peaks = np.maximum.accumulate(with_start, axis=1)[:, 1:]
    max_dd = np.min(running / peaks - 1.0, axis=1)
    daily = returns.ravel()
    losses = -daily
    var95 = float(np.quantile(losses, 0.95))
    es95 = float(losses[losses >= var95].mean())
    return {
        "Mean daily return": float(daily.mean()),
        "Daily volatility": float(daily.std(ddof=1)),
        "VaR 95% loss": var95,
        "Expected shortfall 95%": es95,
        "Final price median": float(np.median(final_prices)),
        "Final price 5th percentile": float(np.quantile(final_prices, 0.05)),
        "Final price 95th percentile": float(np.quantile(final_prices, 0.95)),
        "Probability final price below start": float(np.mean(final_prices < start_price)),
        "Median maximum drawdown": float(np.median(max_dd)),
        "5th percentile maximum drawdown": float(np.quantile(max_dd, 0.05)),
        "Median realised volatility": float(np.median(np.std(returns, axis=1, ddof=1) * np.sqrt(252))),
        "_final_prices": final_prices,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_returns(TRAIN_FILE), load_returns(TEST_FILE)
    train_r = train_df["return"].to_numpy(dtype=float)
    test_r = test_df["return"].to_numpy(dtype=float)
    train_r, test_r = train_r[np.isfinite(train_r)], test_r[np.isfinite(test_r)]
    if len(train_r) < 250 or len(test_r) < 20:
        raise ValueError(f"Not enough data: train={len(train_r)}, test={len(test_r)}")
    start_price = np.nan
    if "close" in train_df.columns:
        closes = train_df["close"].dropna()
        if len(closes):
            start_price = float(closes.iloc[-1])
    if not np.isfinite(start_price):
        start_price = 100.0

    horizon = len(test_r)
    rng = np.random.default_rng(SEED)
    print("FOUR-MODEL MONTE CARLO — dependency-free GARCH")
    print(f"Training observations: {len(train_r)} | Test horizon: {horizon} trading days")
    print(f"Start price: {start_price:.2f} | Monte Carlo paths per model: {N_PATHS}")
    print("\nFitting models on training data only...")

    normal_paths = rng.normal(train_r.mean(), train_r.std(ddof=1), (N_PATHS, horizon)).astype(np.float32)
    raw_hybrid = fit_hybrid(train_r, RAW_TAIL_THRESHOLD)
    hybrid_paths = draw_hybrid(raw_hybrid, (N_PATHS, horizon), rng).astype(np.float32)

    garch = fit_garch11_normal(train_r)
    std_hybrid = draw_standardized_hybrid(garch["std_resid"], STD_TAIL_THRESHOLD)
    cal_rng = np.random.default_rng(SEED + 100)
    calibration = draw_hybrid(std_hybrid, (500_000,), cal_rng)
    std_hybrid["norm_mean"] = float(calibration.mean())
    std_hybrid["norm_sd"] = float(calibration.std(ddof=1))

    garch_normal = garch_paths(garch, N_PATHS, horizon, np.random.default_rng(SEED + 1), "normal")
    garch_hybrid = garch_paths(garch, N_PATHS, horizon, np.random.default_rng(SEED + 2),
                               "hybrid", std_hybrid)

    print("\nGARCH(1,1) fit (returns scaled to percent)")
    print(f"mu={garch['mu']:.6f}, omega={garch['omega']:.6f}, "
          f"alpha={garch['alpha']:.4f}, beta={garch['beta']:.4f}, "
          f"alpha+beta={garch['alpha']+garch['beta']:.4f}")
    print(f"Raw hybrid: threshold={raw_hybrid['threshold']:.2%}, tail n={raw_hybrid['tail_n']}, "
          f"tail probability={raw_hybrid['p_tail']:.2%}, alpha={raw_hybrid['alpha']:.3f}")
    print(f"GARCH residual hybrid: threshold={std_hybrid['threshold']:.2f} SD, "
          f"tail n={std_hybrid['tail_n']}, tail probability={std_hybrid['p_tail']:.2%}, "
          f"Pareto alpha={std_hybrid['alpha']:.3f}")

    models = {
        "Normal IID": normal_paths,
        "Power-law hybrid IID": hybrid_paths,
        "GARCH-normal": garch_normal,
        "GARCH-power-law hybrid": garch_hybrid,
    }
    rows = []
    actual = path_metrics(test_r[None, :], start_price)
    rows.append({
        "Model": "Actual test period",
        "Mean daily return": actual["Mean daily return"],
        "Daily volatility": actual["Daily volatility"],
        "VaR 95% loss": actual["VaR 95% loss"],
        "Expected shortfall 95%": actual["Expected shortfall 95%"],
        "Final price median": float(start_price * np.exp(test_r.sum())),
        "Final price 5th percentile": np.nan,
        "Final price 95th percentile": np.nan,
        "Probability final price below start": float(test_r.sum() < 0),
        "Median maximum drawdown": actual["Median maximum drawdown"],
        "5th percentile maximum drawdown": np.nan,
        "Median realised volatility": float(np.std(test_r, ddof=1) * np.sqrt(252)),
    })
    final_dists = {}
    for name, paths in models.items():
        m = path_metrics(paths, start_price)
        final_dists[name] = m["_final_prices"]
        rows.append({"Model": name, **{k: v for k, v in m.items() if not k.startswith("_")}})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "model_summary.csv", index=False)

    # Save actual and simulated daily log returns for final diagnostics.
    pd.DataFrame({"Actual test return": test_r}).to_csv(
        OUT_DIR / "actual_test_returns.csv", index=False
    )
    for name, paths in models.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        np.save(OUT_DIR / f"{safe_name}_simulated_returns.npy", paths)

    event_rows = []
    for threshold in (0.02, 0.03, 0.05):
        event_rows.append({"Source": "Actual test period", "Threshold": threshold,
                           "Direction": "fall", "Count": int(np.sum(test_r <= -threshold)),
                           "Expected count over horizon": np.nan})
        event_rows.append({"Source": "Actual test period", "Threshold": threshold,
                           "Direction": "rise", "Count": int(np.sum(test_r >= threshold)),
                           "Expected count over horizon": np.nan})
        for name, paths in models.items():
            event_rows.append({"Source": name, "Threshold": threshold, "Direction": "fall",
                               "Count": np.nan, "Expected count over horizon": float(np.mean(paths <= -threshold) * horizon)})
            event_rows.append({"Source": name, "Threshold": threshold, "Direction": "rise",
                               "Count": np.nan, "Expected count over horizon": float(np.mean(paths >= threshold) * horizon)})
    pd.DataFrame(event_rows).to_csv(OUT_DIR / "extreme_event_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    actual_price_path = start_price * np.exp(np.cumsum(test_r))
    axes[0].plot(actual_price_path, label="Actual test path", linewidth=2)
    for name, paths in models.items():
        median_path = np.median(start_price * np.exp(np.cumsum(paths, axis=1)), axis=0)
        axes[0].plot(median_path, label=f"{name} median")
    axes[0].set_title("Actual price vs median simulated paths")
    axes[0].set_xlabel("Trading day in test horizon")
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for name, values in final_dists.items():
        axes[1].hist(values, bins=80, density=True, histtype="step", linewidth=1.3, label=name)
    axes[1].axvline(actual_price_path[-1], linestyle="--", linewidth=2, label="Actual final price")
    axes[1].set_title("Simulated final-price distributions")
    axes[1].set_xlabel("Final price")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "four_model_comparison.png", dpi=160)
    plt.close(fig)

    print("\nMODEL SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults saved in: {OUT_DIR}")
    print("Output files: model_summary.csv, extreme_event_counts.csv, four_model_comparison.png")
    print("\nCaution: thresholds are provisional. This is a risk/scenario model, not a guaranteed price forecast.")
    print("The test period was examined in earlier experiments, so treat this comparison as exploratory.")


if __name__ == "__main__":
    main()
