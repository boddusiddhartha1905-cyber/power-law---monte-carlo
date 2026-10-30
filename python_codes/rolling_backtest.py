#!/usr/bin/env python3
"""
Rolling-origin backtest for four daily-return distribution models:
  1. Normal IID
  2. Power-law hybrid IID
  3. GARCH-normal
  4. GARCH + power-law hybrid standardized innovations

No `arch` package required. Uses NumPy, pandas, SciPy and Matplotlib.
Forecasts one day at a time with expanding training windows. At each origin,
all model parameters are fitted only to data available before that day.

Scores:
  - Log score from predictive density (higher is better; we report mean negative
    log score / NLL, so LOWER is better)
  - CRPS (lower is better)
  - 95% VaR exceedance rate (target about 5%; lower is not automatically better)
  - 95% Expected Shortfall and basic coverage diagnostics

Important: this is a research prototype. Hybrid-tail density is an approximation,
and the power-law thresholds are provisional.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------------- CONFIG -------------------------------
TRAIN_FILE = Path.home() / "reliance_train.csv"
TEST_FILE = Path.home() / "reliance_test.csv"
PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
OUT_DIR = PROJECT_DIR / "rolling_backtest_results"

# To keep runtime reasonable, use expanding-window yearly origins inside the
# available history, then test the current held-out period one day at a time.
# Existing 8y training file is split into initial fit + rolling annual blocks.
INITIAL_TRAIN_FRACTION = 0.60
N_PATHS_CRPS = 2500       # Monte Carlo draws per day for CRPS
SEED = 20261023
RAW_TAIL_THRESHOLD = 0.025
STD_TAIL_THRESHOLD = 2.0
# ----------------------------------------------------------------------


def find_column(df, candidates):
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in cols:
            return cols[key]
    for c in df.columns:
        name = str(c).strip().lower().replace(" ", "_")
        if any(x.strip().lower().replace(" ", "_") in name for x in candidates):
            return c
    return None


def load_returns(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; update file paths at top of script.")
    df = pd.read_csv(path)
    date_col = find_column(df, ["date", "datetime", "timestamp"])
    ret_col = find_column(df, ["log_return", "log_returns", "return", "returns", "daily_return"])
    close_col = find_column(df, ["close", "adj_close", "adj close", "price"])
    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col is not None else None

    if ret_col is not None:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        vals = r.dropna().to_numpy()
        if len(vals) and np.nanpercentile(np.abs(vals), 95) > 0.5:
            warnings.warn(f"{path.name}: returns appear to be percent units; converting to decimals.")
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
    raise ValueError(f"No returns/close column found in {path.name}: {list(df.columns)}")


def fit_hybrid(r, threshold):
    r = np.asarray(r, float)
    mask = np.abs(r) >= threshold
    if int(mask.sum()) < 30:
        raise ValueError(f"Too few tail observations at threshold {threshold}.")
    tail = np.abs(r[mask])
    alpha = len(tail) / np.sum(np.log(tail / threshold))
    return {
        "threshold": threshold, "alpha": float(alpha),
        "p_tail": float(mask.mean()),
        "neg": float(np.mean(r[mask] < 0)),
        "body": r[~mask], "tail_n": int(mask.sum()),
    }


def sample_hybrid(params, n, rng):
    tail = rng.random(n) < params["p_tail"]
    x = np.empty(n)
    nb = int((~tail).sum())
    nt = int(tail.sum())
    if nb:
        x[~tail] = rng.choice(params["body"], size=nb, replace=True)
    if nt:
        u = np.clip(rng.random(nt), 0.0, 1.0 - 1e-12)
        mag = params["threshold"] * (1 - u) ** (-1 / params["alpha"])
        x[tail] = np.where(rng.random(nt) < params["neg"], -mag, mag)
    return x


def hybrid_logpdf(x, params):
    """Approximate density: empirical smoothed body + Pareto signed tails."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    tail = np.abs(x) >= params["threshold"]
    # Tail density: p_tail * alpha*u^alpha / |x|^(alpha+1), split by sign.
    if np.any(tail):
        signprob = np.where(x[tail] < 0, params["neg"], 1 - params["neg"])
        density = params["p_tail"] * signprob * params["alpha"] * params["threshold"]**params["alpha"]
        density = density / np.maximum(np.abs(x[tail]), params["threshold"])**(params["alpha"] + 1)
        out[tail] = np.log(np.maximum(density, 1e-300))
    # Body density via Gaussian kernel estimate with bandwidth selected by Silverman rule.
    if np.any(~tail):
        body = params["body"]
        sd = max(float(np.std(body, ddof=1)), 1e-6)
        bw = max(1.06 * sd * max(len(body), 2) ** (-0.2), 1e-5)
        dif = (x[~tail, None] - body[None, :]) / bw
        # Chunking isn't needed here: number of test observations is modest.
        vals = np.exp(-0.5 * dif**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
        # Include the body-mixture probability so the full density integrates correctly.
        vals = (1.0 - params["p_tail"]) * vals
        out[~tail] = np.log(np.maximum(vals, 1e-300))
    return out


def fit_garch(r):
    """Gaussian QMLE for GARCH(1,1), return scale is percentage points."""
    y = np.asarray(r, float) * 100
    mu0 = float(np.mean(y))
    v0 = max(float(np.var(y - mu0)), 1e-6)

    def unpack(theta):
        omega = np.exp(np.clip(theta[0], -20, 10))
        alpha = 1 / (1 + np.exp(-np.clip(theta[1], -20, 20)))
        beta_raw = 1 / (1 + np.exp(-np.clip(theta[2], -20, 20)))
        beta = beta_raw * max(1e-8, 0.999 - alpha)
        return float(theta[3]), float(omega), float(alpha), float(beta)

    def nll(theta):
        mu, omega, alpha, beta = unpack(theta)
        eps = y - mu
        h = np.empty_like(y)
        h[0] = max(v0, omega / max(1e-5, 1 - alpha - beta))
        for i in range(1, len(y)):
            h[i] = omega + alpha * eps[i - 1]**2 + beta * h[i - 1]
        h = np.maximum(h, 1e-10)
        return float(0.5 * np.sum(np.log(2*np.pi) + np.log(h) + eps**2/h))

    init = np.array([np.log(max(v0 * 0.05, 1e-5)), -1.5, 2.0, mu0])
    opt = minimize(nll, init, method="Nelder-Mead",
                   options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-6})
    if not opt.success:
        warnings.warn(f"GARCH optimizer: {opt.message}")
    mu, omega, alpha, beta = unpack(opt.x)
    eps = y - mu
    h = np.empty_like(y)
    h[0] = max(v0, omega / max(1e-5, 1 - alpha - beta))
    for i in range(1, len(y)):
        h[i] = omega + alpha * eps[i - 1]**2 + beta * h[i - 1]
    z = eps / np.sqrt(np.maximum(h, 1e-10))
    z = (z - z.mean()) / max(float(z.std(ddof=1)), 1e-8)
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
            "eps_last": float(eps[-1]), "var_last": float(h[-1]), "z": z}


def normal_nll(x, mu, sd):
    return -norm.logpdf(x, loc=mu, scale=max(sd, 1e-8))


def crps_ensemble(samples, observation):
    """CRPS = E|X-y| - 0.5 E|X-X'|, empirical sample estimate."""
    x = np.asarray(samples, float)
    first = np.mean(np.abs(x - observation))
    xs = np.sort(x)
    n = len(xs)
    # E|X-X'| / 2 = sum((2i-n-1)*x_i)/n^2, i=1..n
    idx = np.arange(1, n + 1)
    half_pair = np.sum((2 * idx - n - 1) * xs) / (n * n)
    return float(first - half_pair)


def simulate_garch_one_step(g, n, rng, hybrid=None):
    var_next = g["omega"] + g["alpha"] * g["eps_last"]**2 + g["beta"] * g["var_last"]
    var_next = max(float(var_next), 1e-10)
    if hybrid is None:
        z = rng.standard_normal(n)
    else:
        z = sample_hybrid(hybrid, n, rng)
        # Standardize innovation distribution using training-only sample approximation.
        z = (z - hybrid["norm_mean"]) / hybrid["norm_sd"]
    return (g["mu"] + np.sqrt(var_next) * z) / 100.0, var_next


def score_window(train, observations, model_names, rng):
    """Sequential one-step scores; re-fit every 20 observations for tractable runtime."""
    rows = []
    prev_train = np.asarray(train, float).copy()
    # To avoid excessive optimizer calls, refit every 20 test days, but update
    # last residual/variance recursively each day. No future observation is used.
    fit_cache = None
    cache_age = 20
    garch_state = None
    for j, obs in enumerate(np.asarray(observations, float)):
        if j % 20 == 0 or fit_cache is None:
            mu = float(prev_train.mean())
            sd = max(float(prev_train.std(ddof=1)), 1e-6)
            raw_hybrid = fit_hybrid(prev_train, RAW_TAIL_THRESHOLD)
            garch_state = fit_garch(prev_train)
            zhybrid = fit_hybrid(garch_state["z"], STD_TAIL_THRESHOLD)
            # Estimate innovation hybrid mean/SD with training-only Monte Carlo.
            cal = sample_hybrid(zhybrid, 100_000, rng)
            zhybrid["norm_mean"] = float(cal.mean())
            zhybrid["norm_sd"] = max(float(cal.std(ddof=1)), 1e-6)
            fit_cache = (mu, sd, raw_hybrid, garch_state, zhybrid)
        mu, sd, raw_hybrid, g, zhybrid = fit_cache

        # Normal IID predictive distribution.
        ns = rng.normal(mu, sd, N_PATHS_CRPS)
        # Hybrid IID predictive distribution.
        hs = sample_hybrid(raw_hybrid, N_PATHS_CRPS, rng)
        # GARCH one-step predictive distributions.
        gn, _ = simulate_garch_one_step(g, N_PATHS_CRPS, rng)
        gh, _ = simulate_garch_one_step(g, N_PATHS_CRPS, rng, zhybrid)

        # Density scores (NLL lower is better).
        scores = {
            "Normal IID": (float(normal_nll(obs, mu, sd)), crps_ensemble(ns, obs),
                           float(np.mean(ns <= np.quantile(ns, 0.05)))),
            "Power-law hybrid IID": (float(-hybrid_logpdf(np.array([obs]), raw_hybrid)[0]),
                                     crps_ensemble(hs, obs), float(np.mean(hs <= np.quantile(hs, .05)))),
            "GARCH-normal": (float(-norm.logpdf(obs, loc=g["mu"]/100,
                                                 scale=np.sqrt(g["omega"] + g["alpha"]*g["eps_last"]**2 + g["beta"]*g["var_last"])/100)),
                             crps_ensemble(gn, obs), float(np.mean(gn <= np.quantile(gn, .05)))),
            "GARCH-power-law hybrid": (float(-hybrid_logpdf(np.array([(obs*100-g["mu"]) /
                                        np.sqrt(g["omega"] + g["alpha"]*g["eps_last"]**2 + g["beta"]*g["var_last"])]),
                                        zhybrid)[0] - np.log(100 / np.sqrt(g["omega"] + g["alpha"]*g["eps_last"]**2 + g["beta"]*g["var_last"]))),
                                      crps_ensemble(gh, obs), float(np.mean(gh <= np.quantile(gh, .05)))),
        }
        # Empirical one-day lower 5% quantile for VaR exceedance.
        for name, (nll, crps, _) in scores.items():
            if name == "Normal IID":
                draws = ns
            elif name == "Power-law hybrid IID":
                draws = hs
            elif name == "GARCH-normal":
                draws = gn
            else:
                draws = gh
            q05 = float(np.quantile(draws, 0.05))
            rows.append({"day_index": j, "Model": name, "actual_return": obs,
                         "negative_log_score": nll, "CRPS": crps,
                         "VaR95_loss": -q05, "VaR95_exceedance": int(obs < q05)})
        # After scoring today's observation, it may be added for the next forecast.
        prev_train = np.append(prev_train, obs)
        # Update the GARCH state for next day using the observation, but keep fitted params
        # fixed until next refit. This is strictly sequential.
        eps_now = obs * 100 - g["mu"]
        var_now = g["omega"] + g["alpha"] * g["eps_last"]**2 + g["beta"] * g["var_last"]
        g["eps_last"] = float(eps_now)
        g["var_last"] = float(max(var_now, 1e-10))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_returns(TRAIN_FILE)
    test_df = load_returns(TEST_FILE)
    train = train_df["return"].to_numpy(float)
    test = test_df["return"].to_numpy(float)
    train, test = train[np.isfinite(train)], test[np.isfinite(test)]
    if len(train) < 500 or len(test) < 50:
        raise ValueError(f"Not enough observations: train={len(train)}, test={len(test)}")

    # Rolling validation inside the training file: expanding-window, contiguous folds.
    initial_n = int(len(train) * INITIAL_TRAIN_FRACTION)
    # Use 4 contiguous blocks from the remainder of the training period.
    remaining = len(train) - initial_n
    block = max(40, remaining // 4)
    rng = np.random.default_rng(SEED)
    fold_frames = []
    start = initial_n
    fold_id = 1
    while start < len(train):
        end = min(start + block, len(train))
        if end - start < 40:
            break
        scored = score_window(train[:start], train[start:end], None, rng)
        scored["evaluation_period"] = "within_training"
        scored["fold"] = fold_id
        scored["train_n"] = start
        fold_frames.append(scored)
        start = end
        fold_id += 1

    # Final exploratory sequential evaluation on supplied test file.
    test_scored = score_window(train, test, None, rng)
    test_scored["evaluation_period"] = "existing_test_exploratory"
    test_scored["fold"] = 0
    test_scored["train_n"] = len(train)
    all_scores = pd.concat(fold_frames + [test_scored], ignore_index=True)
    all_scores.to_csv(OUT_DIR / "daily_scores.csv", index=False)

    summary = (all_scores.groupby(["evaluation_period", "Model"], as_index=False)
               .agg(days=("actual_return", "size"),
                    mean_negative_log_score=("negative_log_score", "mean"),
                    mean_CRPS=("CRPS", "mean"),
                    VaR95_exceedances=("VaR95_exceedance", "sum"),
                    VaR95_expected_exceedances=("VaR95_exceedance", lambda x: len(x) * .05),
                    VaR95_exceedance_rate=("VaR95_exceedance", "mean"),
                    mean_VaR95_loss=("VaR95_loss", "mean")))
    summary.to_csv(OUT_DIR / "score_summary.csv", index=False)

    fold_summary = (all_scores.groupby(["evaluation_period", "fold", "Model"], as_index=False)
                    .agg(days=("actual_return", "size"),
                         mean_negative_log_score=("negative_log_score", "mean"),
                         mean_CRPS=("CRPS", "mean"),
                         VaR95_exceedance_rate=("VaR95_exceedance", "mean")))
    fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_data = summary[summary["evaluation_period"] == "within_training"]
    if len(plot_data):
        axes[0].bar(plot_data["Model"], plot_data["mean_CRPS"])
        axes[0].set_title("Within-training rolling validation: mean CRPS")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set_ylabel("CRPS (lower is better)")
    test_summary = summary[summary["evaluation_period"] == "existing_test_exploratory"]
    if len(test_summary):
        axes[1].bar(test_summary["Model"], test_summary["mean_negative_log_score"])
        axes[1].set_title("Existing test period: mean negative log score")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_ylabel("Negative log score (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_comparison.png", dpi=160)
    plt.close(fig)

    print("\nROLLING BACKTEST COMPLETE")
    print(f"Training observations: {len(train)} | Existing test observations: {len(test)}")
    print(f"Within-training rolling folds: {len(fold_frames)}")
    print("\nSCORE SUMMARY (lower CRPS and negative log score are better; VaR rate should be near 5%)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved to: {OUT_DIR}")
    print("Files: score_summary.csv, daily_scores.csv, fold_summary.csv, score_comparison.png")
    print("\nCaution: thresholds are provisional, and the existing test period was previously examined.")
    print("Compare results across folds; do not select a model from one metric or one period alone.")


if __name__ == "__main__":
    main()

