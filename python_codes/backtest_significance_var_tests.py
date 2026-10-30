#!/usr/bin/env python3
"""
Paired block-bootstrap significance tests for CRPS and formal VaR backtests.

Input:
  ~/Documents/power law - monte carlo/rolling_backtest_results/daily_scores.csv

Outputs:
  significance_var_results/crps_block_bootstrap.csv
  significance_var_results/var_backtests.csv

Tests:
  1) Paired moving-block bootstrap confidence interval for mean CRPS difference.
     Difference = competitor CRPS - GARCH-power-law hybrid CRPS; positive favors
     GARCH-power-law hybrid. Uses contiguous day blocks to partly preserve dependence.
  2) Kupiec unconditional coverage test for 5% lower-tail VaR.
  3) Christoffersen independence test and conditional-coverage test.

Caution:
  - Daily scores are produced by the prior rolling_backtest.py implementation.
  - Its VaR indicators are based on Monte Carlo forecast quantiles and can have
    simulation noise. Treat results as diagnostics, not a final research claim.
  - Four validation folds / one previously examined test period are limited evidence.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
INPUT_FILE = PROJECT_DIR / "rolling_backtest_results" / "daily_scores.csv"
OUT_DIR = PROJECT_DIR / "significance_var_results"

BENCHMARK = "GARCH-power-law hybrid"
BLOCK_LENGTH = 10
BOOTSTRAP_REPS = 5000
ALPHA = 0.05
SEED = 20261023


def log_binom_likelihood(n, x, p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    x = int(x)
    n = int(n)
    return x * np.log(p) + (n - x) * np.log(1 - p)


def kupiec_test(hits, p=0.05):
    hits = np.asarray(hits, dtype=int)
    n = len(hits)
    x = int(hits.sum())
    phat = x / n if n else np.nan
    if n == 0:
        return {"n": 0, "exceptions": 0, "exception_rate": np.nan,
                "LR_uc": np.nan, "p_uc": np.nan}
    ll_null = log_binom_likelihood(n, x, p)
    ll_alt = log_binom_likelihood(n, x, phat) if 0 < x < n else 0.0
    lr = max(0.0, -2 * (ll_null - ll_alt))
    return {"n": n, "exceptions": x, "exception_rate": phat,
            "LR_uc": lr, "p_uc": float(chi2.sf(lr, 1))}


def christoffersen_test(hits):
    """First-order Markov independence test for VaR exceptions."""
    h = np.asarray(hits, dtype=int)
    if len(h) < 3:
        return {"LR_ind": np.nan, "p_ind": np.nan, "LR_cc": np.nan, "p_cc": np.nan}
    h0, h1 = h[:-1], h[1:]
    n00 = int(np.sum((h0 == 0) & (h1 == 0)))
    n01 = int(np.sum((h0 == 0) & (h1 == 1)))
    n10 = int(np.sum((h0 == 1) & (h1 == 0)))
    n11 = int(np.sum((h0 == 1) & (h1 == 1)))
    d0 = n00 + n01
    d1 = n10 + n11
    pi0 = n01 / d0 if d0 else 0.0
    pi1 = n11 / d1 if d1 else 0.0
    pi = (n01 + n11) / max(d0 + d1, 1)

    def bern_ll(success, total, prob):
        if total == 0:
            return 0.0
        prob = float(np.clip(prob, 1e-12, 1 - 1e-12))
        return success * np.log(prob) + (total - success) * np.log(1 - prob)

    ll_ind_null = bern_ll(n01 + n11, d0 + d1, pi)
    ll_ind_alt = bern_ll(n01, d0, pi0) + bern_ll(n11, d1, pi1)
    lr_ind = max(0.0, -2 * (ll_ind_null - ll_ind_alt))
    p_ind = float(chi2.sf(lr_ind, 1))

    uc = kupiec_test(h)
    lr_cc = uc["LR_uc"] + lr_ind
    p_cc = float(chi2.sf(lr_cc, 2))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "LR_ind": lr_ind, "p_ind": p_ind,
            "LR_cc": lr_cc, "p_cc": p_cc}


def moving_block_bootstrap_mean(diff, block_len, reps, rng):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    starts = np.arange(n)
    boot_means = np.empty(reps)
    for b in range(reps):
        picked = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([diff[(s + np.arange(block_len)) % n] for s in picked])[:n]
        boot_means[b] = sample.mean()
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run rolling_backtest.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    required = {"evaluation_period", "Model", "day_index", "CRPS", "VaR95_exceedance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"daily_scores.csv missing columns: {sorted(missing)}")

    rng = np.random.default_rng(SEED)
    sig_rows = []
    var_rows = []
    periods = list(df["evaluation_period"].dropna().unique())

    for period in periods:
        sub = df[df["evaluation_period"] == period].copy()
        wide_crps = sub.pivot_table(index="day_index", columns="Model", values="CRPS", aggfunc="mean").sort_index()
        if BENCHMARK not in wide_crps.columns:
            raise ValueError(f"Benchmark '{BENCHMARK}' missing from {period}")
        for model in wide_crps.columns:
            if model == BENCHMARK:
                continue
            paired = wide_crps[[model, BENCHMARK]].dropna()
            # Positive means competitor has larger CRPS, so benchmark is better.
            diff = paired[model].to_numpy() - paired[BENCHMARK].to_numpy()
            mean_diff, lo, hi = moving_block_bootstrap_mean(diff, BLOCK_LENGTH, BOOTSTRAP_REPS, rng)
            sig_rows.append({
                "evaluation_period": period,
                "comparison": f"{BENCHMARK} vs {model}",
                "paired_days": len(diff),
                "block_length_days": min(BLOCK_LENGTH, len(diff)),
                "mean_CRPS_advantage_of_benchmark": mean_diff,
                "bootstrap_95CI_low": lo,
                "bootstrap_95CI_high": hi,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "interpretation": (
                    "Benchmark lower CRPS; CI excludes zero" if lo > 0 else
                    "Competitor lower CRPS; CI excludes zero" if hi < 0 else
                    "Difference not clearly distinguishable from zero"
                ),
            })

        # VaR tests for each model and evaluation period.
        for model in sorted(sub["Model"].unique()):
            m = sub[sub["Model"] == model].sort_values("day_index")
            hits = m["VaR95_exceedance"].fillna(0).astype(int).to_numpy()
            uc = kupiec_test(hits, ALPHA)
            ind = christoffersen_test(hits)
            var_rows.append({
                "evaluation_period": period,
                "Model": model,
                **uc,
                **ind,
                "target_exception_rate": ALPHA,
                "coverage_test_reject_5pct": bool(uc["p_uc"] < ALPHA) if np.isfinite(uc["p_uc"]) else False,
                "independence_test_reject_5pct": bool(ind["p_ind"] < ALPHA) if np.isfinite(ind["p_ind"]) else False,
                "conditional_coverage_reject_5pct": bool(ind["p_cc"] < ALPHA) if np.isfinite(ind["p_cc"]) else False,
            })

    sig = pd.DataFrame(sig_rows)
    var = pd.DataFrame(var_rows)
    sig.to_csv(OUT_DIR / "crps_block_bootstrap.csv", index=False)
    var.to_csv(OUT_DIR / "var_backtests.csv", index=False)

    print("CRPS PAIRED MOVING-BLOCK BOOTSTRAP")
    print("Positive mean advantage favors GARCH-power-law hybrid; CI crossing zero means inconclusive.")
    print(sig.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFORMAL 95% VaR BACKTESTS")
    print("Coverage target is 5%. Small p-values indicate evidence against correct coverage/independence.")
    show_cols = ["evaluation_period", "Model", "n", "exceptions", "exception_rate",
                 "p_uc", "p_ind", "p_cc", "coverage_test_reject_5pct",
                 "independence_test_reject_5pct", "conditional_coverage_reject_5pct"]
    print(var[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to: {OUT_DIR}")
    print("Files: crps_block_bootstrap.csv, var_backtests.csv")
    print("\nCaution: these tests inherit the upstream scoring/forecast implementation and are exploratory.")
    print("Do not interpret failure to reject as proof that a model is correct.")


if __name__ == "__main__":
    main()
