#!/usr/bin/env python3
"""Four-model Monte Carlo comparison for Titan and Jio Financial.

Models: Normal IID, power-law hybrid IID, GARCH-normal, GARCH-power-law hybrid.
Parameters are fitted on each asset's training data only. No Student-t distribution.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
DATA_DIR = PROJECT_DIR / "asset_data"
ASSETS = {
    "titan": ("titan_train.csv", "titan_test.csv"),
    "jio_financial": ("jio_financial_train.csv", "jio_financial_test.csv"),
}
N_PATHS = 10_000
SEED = 42
RAW_TAIL_THRESHOLD = 0.025
STD_TAIL_THRESHOLD = 2.0

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
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    date_col = find_column(df, ["date", "datetime", "timestamp"])
    ret_col = find_column(df, ["log_return", "log_returns", "return", "returns", "daily_return"])
    close_col = find_column(df, ["close", "adj_close", "adj close", "price"])
    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col is not None else None
    if ret_col is not None:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        vals = r.dropna().to_numpy()
        if len(vals) and np.nanpercentile(np.abs(vals), 95) > 0.5:
            warnings.warn(f"{path.name}: returns appear in percent units; converting to decimals.")
            r = r / 100.0
        out = pd.DataFrame({"return": r})
        if dates is not None: out["date"] = dates
        if close_col is not None: out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        return out.dropna(subset=["return"]).reset_index(drop=True)
    if close_col is not None:
        close = pd.to_numeric(df[close_col], errors="coerce")
        out = pd.DataFrame({"return": np.log(close / close.shift(1)), "close": close})
        if dates is not None: out["date"] = dates
        return out.dropna(subset=["return"]).reset_index(drop=True)
    raise ValueError(f"No return/close column found in {path.name}")

def fit_hybrid(r, threshold):
    r = np.asarray(r, float)
    mask = np.abs(r) >= threshold
    if int(mask.sum()) < 30:
        raise ValueError(f"Too few tail observations at threshold {threshold}; found {mask.sum()}.")
    tail = np.abs(r[mask])
    alpha = len(tail) / np.sum(np.log(tail / threshold))
    return {"threshold": float(threshold), "alpha": float(alpha), "p_tail": float(mask.mean()),
            "tail_sign_negative": float(np.mean(r[mask] < 0)), "body": r[~mask],
            "tail_n": int(mask.sum())}

def draw_hybrid(params, shape, rng):
    n = int(np.prod(shape))
    tail = rng.random(n) < params["p_tail"]
    out = np.empty(n, float)
    nb, nt = int((~tail).sum()), int(tail.sum())
    if nb: out[~tail] = rng.choice(params["body"], size=nb, replace=True)
    if nt:
        u = np.clip(rng.random(nt), 0, 1 - 1e-12)
        mag = params["threshold"] * (1-u) ** (-1/params["alpha"])
        out[tail] = np.where(rng.random(nt) < params["tail_sign_negative"], -mag, mag)
    return out.reshape(shape)

def fit_garch11_normal(train_returns):
    y = np.asarray(train_returns, float) * 100
    mu0 = float(np.mean(y))
    v0 = max(float(np.var(y-mu0)), 1e-6)
    def unpack(theta):
        omega = np.exp(np.clip(theta[0], -20, 10))
        alpha = 1/(1+np.exp(-np.clip(theta[1], -20, 20)))
        beta_raw = 1/(1+np.exp(-np.clip(theta[2], -20, 20)))
        beta = beta_raw * max(1e-8, 0.999-alpha)
        return float(theta[3]), float(omega), float(alpha), float(beta)
    def nll(theta):
        mu, omega, alpha, beta = unpack(theta)
        eps = y-mu
        h = np.empty_like(y)
        h[0] = max(v0, omega/max(1e-5, 1-alpha-beta))
        for t in range(1, len(y)):
            h[t] = omega + alpha*eps[t-1]**2 + beta*h[t-1]
        h = np.maximum(h, 1e-10)
        return float(.5*np.sum(np.log(2*np.pi)+np.log(h)+eps**2/h))
    start = np.array([np.log(max(v0*.05, 1e-5)), -1.5, 2.0, mu0])
    result = minimize(nll, start, method="Nelder-Mead",
                      options={"maxiter": 12000, "xatol": 1e-7, "fatol": 1e-7})
    if not result.success: warnings.warn(f"GARCH optimizer note: {result.message}")
    mu, omega, alpha, beta = unpack(result.x)
    eps = y-mu
    h = np.empty_like(y)
    h[0] = max(v0, omega/max(1e-5, 1-alpha-beta))
    for t in range(1, len(y)): h[t] = omega+alpha*eps[t-1]**2+beta*h[t-1]
    h = np.maximum(h, 1e-10)
    z = eps/np.sqrt(h)
    z = (z-z.mean())/max(float(z.std(ddof=1)), 1e-8)
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
            "last_eps": float(eps[-1]), "last_var": float(h[-1]), "std_resid": z}

def garch_paths(params, n_paths, horizon, rng, innovation="normal", innovation_params=None):
    mu, omega, alpha, beta = params["mu"], params["omega"], params["alpha"], params["beta"]
    eps_prev = np.full(n_paths, params["last_eps"])
    var_prev = np.full(n_paths, params["last_var"])
    out = np.empty((n_paths, horizon), dtype=np.float32)
    for t in range(horizon):
        var_t = np.maximum(omega+alpha*eps_prev**2+beta*var_prev, 1e-12)
        if innovation == "normal":
            z = rng.standard_normal(n_paths)
        else:
            z = draw_hybrid(innovation_params, (n_paths,), rng)
            z = (z-innovation_params["norm_mean"])/innovation_params["norm_sd"]
        eps_t = np.sqrt(var_t)*z
        out[:, t] = (mu+eps_t)/100
        eps_prev, var_prev = eps_t, var_t
    return out

def path_metrics(returns, start_price):
    final_prices = start_price*np.exp(np.sum(returns, axis=1))
    running = start_price*np.exp(np.cumsum(returns, axis=1))
    with_start = np.column_stack([np.full(len(returns), start_price), running])
    peaks = np.maximum.accumulate(with_start, axis=1)[:, 1:]
    max_dd = np.min(running/peaks-1, axis=1)
    daily = returns.ravel()
    losses = -daily
    var95 = float(np.quantile(losses, .95))
    es95 = float(losses[losses >= var95].mean())
    return {
        "Mean daily return": float(daily.mean()), "Daily volatility": float(daily.std(ddof=1)),
        "VaR 95% loss": var95, "Expected shortfall 95%": es95,
        "Final price median": float(np.median(final_prices)),
        "Final price 5th percentile": float(np.quantile(final_prices, .05)),
        "Final price 95th percentile": float(np.quantile(final_prices, .95)),
        "Probability final price below start": float(np.mean(final_prices < start_price)),
        "Median maximum drawdown": float(np.median(max_dd)),
        "5th percentile maximum drawdown": float(np.quantile(max_dd, .05)),
        "Median realised volatility": float(np.median(np.std(returns, axis=1, ddof=1)*np.sqrt(252))),
        "_final_prices": final_prices}

def run_asset(asset, files):
    train_file, test_file = files
    out_dir = PROJECT_DIR / f"{asset}_four_model_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_returns(DATA_DIR/train_file)
    test_df = load_returns(DATA_DIR/test_file)
    train = train_df["return"].to_numpy(float); test = test_df["return"].to_numpy(float)
    train, test = train[np.isfinite(train)], test[np.isfinite(test)]
    if len(train)<250 or len(test)<20: raise ValueError(f"{asset}: insufficient train/test data")
    # Test files contain log returns only. Use the final adjusted price in the raw file
    # as the test period's starting price so simulated price levels have a real anchor.
    raw_name = "titan_10y.csv" if asset=="titan" else "jio_financial_10y.csv"
    raw = pd.read_csv(DATA_DIR/raw_name)
    raw_date = find_column(raw, ["date"])
    price_col = find_column(raw, ["adj_close", "adj close", "close"])
    raw[raw_date] = pd.to_datetime(raw[raw_date], errors="coerce")
    test_dates = pd.to_datetime(test_df["date"], errors="coerce") if "date" in test_df else None
    if test_dates is not None and test_dates.notna().any():
        first_test_date = test_dates.iloc[0]
        earlier = raw.loc[raw[raw_date] < first_test_date].dropna(subset=[price_col])
        start_price = float(pd.to_numeric(earlier[price_col], errors="coerce").dropna().iloc[-1])
    else:
        start_price = 100.0
    horizon = len(test)
    print(f"\n{'='*72}\n{asset.upper()} — FOUR-MODEL MONTE CARLO\n{'='*72}")
    print(f"Train n={len(train)} | test horizon={horizon} | starting price={start_price:.4f} | paths={N_PATHS}")
    rng = np.random.default_rng(SEED)
    normal_paths = rng.normal(train.mean(), train.std(ddof=1), (N_PATHS, horizon)).astype(np.float32)
    raw_hybrid = fit_hybrid(train, RAW_TAIL_THRESHOLD)
    hybrid_paths = draw_hybrid(raw_hybrid, (N_PATHS, horizon), rng).astype(np.float32)
    garch = fit_garch11_normal(train)
    std_hybrid = fit_hybrid(garch["std_resid"], STD_TAIL_THRESHOLD)
    calibration = draw_hybrid(std_hybrid, (300_000,), np.random.default_rng(SEED+100))
    std_hybrid["norm_mean"] = float(calibration.mean())
    std_hybrid["norm_sd"] = max(float(calibration.std(ddof=1)), 1e-6)
    garch_normal = garch_paths(garch, N_PATHS, horizon, np.random.default_rng(SEED+1))
    garch_hybrid = garch_paths(garch, N_PATHS, horizon, np.random.default_rng(SEED+2),
                               "hybrid", std_hybrid)
    print(f"GARCH: mu={garch['mu']:.5f}, omega={garch['omega']:.5f}, "
          f"alpha={garch['alpha']:.4f}, beta={garch['beta']:.4f}, alpha+beta={garch['alpha']+garch['beta']:.4f}")
    print(f"Raw hybrid: tail n={raw_hybrid['tail_n']}, p_tail={raw_hybrid['p_tail']:.3%}, alpha={raw_hybrid['alpha']:.3f}")
    print(f"Std-residual hybrid: tail n={std_hybrid['tail_n']}, p_tail={std_hybrid['p_tail']:.3%}, alpha={std_hybrid['alpha']:.3f}")
    models = {"Normal IID":normal_paths, "Power-law hybrid IID":hybrid_paths,
              "GARCH-normal":garch_normal, "GARCH-power-law hybrid":garch_hybrid}
    rows=[]; finals={}
    actual = path_metrics(test[None,:], start_price)
    rows.append({"Model":"Actual test period", **{k:v for k,v in actual.items() if not k.startswith("_")}})
    for name, paths in models.items():
        m=path_metrics(paths,start_price); finals[name]=m["_final_prices"]
        rows.append({"Model":name, **{k:v for k,v in m.items() if not k.startswith("_")}})
    summary=pd.DataFrame(rows)
    summary.to_csv(out_dir/"model_summary.csv",index=False)

    # Save actual and simulated daily log returns.
    pd.DataFrame({"Actual test return": test}).to_csv(
        out_dir / "actual_test_returns.csv", index=False
    )
    for name, paths in models.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        np.save(out_dir / f"{safe_name}_simulated_returns.npy", paths)
    event_rows=[]
    for threshold in (.02,.03,.05):
        event_rows.append({"Source":"Actual test period","Threshold":threshold,"Direction":"fall",
                           "Count":int(np.sum(test<=-threshold)),"Expected count over horizon":np.nan})
        event_rows.append({"Source":"Actual test period","Threshold":threshold,"Direction":"rise",
                           "Count":int(np.sum(test>=threshold)),"Expected count over horizon":np.nan})
        for name,paths in models.items():
            event_rows.append({"Source":name,"Threshold":threshold,"Direction":"fall","Count":np.nan,
                               "Expected count over horizon":float(np.mean(paths<=-threshold)*horizon)})
            event_rows.append({"Source":name,"Threshold":threshold,"Direction":"rise","Count":np.nan,
                               "Expected count over horizon":float(np.mean(paths>=threshold)*horizon)})
    pd.DataFrame(event_rows).to_csv(out_dir/"extreme_event_counts.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    actual_path=start_price*np.exp(np.cumsum(test))
    axes[0].plot(actual_path,label="Actual test path",linewidth=2)
    for name,paths in models.items():
        med=np.median(start_price*np.exp(np.cumsum(paths,axis=1)),axis=0)
        axes[0].plot(med,label=f"{name} median")
    axes[0].set_title(f"{asset}: actual vs simulated median paths"); axes[0].set_xlabel("Test trading day")
    axes[0].set_ylabel("Price"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.25)
    for name,values in finals.items(): axes[1].hist(values,bins=80,density=True,histtype="step",label=name)
    axes[1].axvline(actual_path[-1],linestyle="--",linewidth=2,label="Actual final price")
    axes[1].set_title(f"{asset}: final-price distributions"); axes[1].set_xlabel("Final price")
    axes[1].set_ylabel("Density"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir/"four_model_comparison.png",dpi=160); plt.close(fig)
    print("\nSUMMARY (lower VaR/ES is not inherently better; compare calibration and CRPS/backtests too)")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("Saved:",out_dir)

def main():
    for asset,files in ASSETS.items():
        run_asset(asset,files)
    print("\nBoth Monte Carlo comparisons finished. Results are in separate project folders.")
    print("Reminder: test period was previously inspected; treat results as exploratory.")

if __name__=="__main__":
    main()
