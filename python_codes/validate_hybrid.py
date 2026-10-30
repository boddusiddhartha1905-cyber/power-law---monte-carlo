
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = Path.home() / "reliance_train.csv"

# Candidate thresholds: training data only.
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

# Avoid selecting a threshold supported by very few tail observations.
MIN_TAIL_OBSERVATIONS = 50

# Bootstrap repetitions. Increase for more precise results if needed.
BOOTSTRAP_REPS = 2000
SEED = 42


# ============================================================
# 2. LOAD TRAINING DATA ONLY
# ============================================================

data = pd.read_csv(DATA_FILE)

if "Log_Return" not in data.columns:
    raise ValueError("The CSV must contain a 'Log_Return' column.")

returns = data["Log_Return"].dropna().to_numpy(dtype=float)

if len(returns) == 0 or not np.all(np.isfinite(returns)):
    raise ValueError("Training returns are empty or contain invalid values.")

rng = np.random.default_rng(SEED)


# ============================================================
# 3. POWER-LAW FIT AND KS DISTANCE
# ============================================================

def fit_pareto_tail(magnitudes, threshold):
    """
    Fit a continuous Pareto tail above threshold.

    Survival function:
        P(X >= x | X >= threshold) = (threshold / x)**alpha

    MLE:
        alpha = n / sum(log(X / threshold))
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    tail = magnitudes[magnitudes >= threshold]

    if len(tail) == 0:
        raise ValueError("No observations exceed the threshold.")

    log_ratios = np.log(tail / threshold)
    denominator = np.sum(log_ratios)

    if denominator <= 0:
        raise ValueError("Cannot estimate alpha for this tail.")

    alpha = len(tail) / denominator
    return alpha, tail


def pareto_ks_distance(tail, threshold, alpha):
    """
    Two-sided one-sample KS distance between the empirical
    conditional tail CDF and the fitted Pareto CDF.
    """
    tail = np.sort(np.asarray(tail, dtype=float))
    n = len(tail)

    if n == 0:
        return np.nan

    fitted_cdf = 1.0 - (threshold / tail) ** alpha
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(0, n) / n

    d_plus = np.max(empirical_upper - fitted_cdf)
    d_minus = np.max(fitted_cdf - empirical_lower)

    return max(d_plus, d_minus)


# ============================================================
# 4. BOOTSTRAP EACH CANDIDATE THRESHOLD
# ============================================================

def evaluate_threshold(threshold):
    magnitudes = np.abs(returns)
    tail = magnitudes[magnitudes >= threshold]
    n_tail = len(tail)

    if n_tail < 2:
        return None

    alpha, tail = fit_pareto_tail(tail, threshold)
    observed_ks = pareto_ks_distance(tail, threshold, alpha)

    alpha_boot = np.empty(BOOTSTRAP_REPS)
    ks_boot = np.empty(BOOTSTRAP_REPS)

    for b in range(BOOTSTRAP_REPS):
        # Parametric bootstrap under the fitted Pareto model.
        u = rng.random(n_tail)
        simulated_tail = threshold * (1.0 - u) ** (-1.0 / alpha)

        alpha_b, _ = fit_pareto_tail(simulated_tail, threshold)
        alpha_boot[b] = alpha_b

        ks_boot[b] = pareto_ks_distance(
            simulated_tail, threshold, alpha_b
        )

    ci_low, ci_high = np.quantile(alpha_boot, [0.025, 0.975])

    # Approximate fixed-threshold goodness-of-fit p-value.
    # This does NOT account for searching over many thresholds.
    gof_p = (
        1 + np.sum(ks_boot >= observed_ks)
    ) / (BOOTSTRAP_REPS + 1)

    return {
        "threshold": threshold,
        "n_tail": n_tail,
        "tail_fraction": n_tail / len(returns),
        "alpha": alpha,
        "alpha_ci_low": ci_low,
        "alpha_ci_high": ci_high,
        "ks": observed_ks,
        "gof_p": gof_p,
        "eligible": n_tail >= MIN_TAIL_OBSERVATIONS,
    }


# ============================================================
# 5. RUN THE TRAINING-ONLY DIAGNOSTICS
# ============================================================

results = []

print("=" * 88)
print("POWER-LAW THRESHOLD AND EXPONENT VALIDATION")
print("=" * 88)
print(f"Training observations: {len(returns)}")
print(f"Minimum tail observations for threshold eligibility: "
      f"{MIN_TAIL_OBSERVATIONS}")
print(f"Bootstrap repetitions per threshold: {BOOTSTRAP_REPS}")
print("The test dataset is not loaded or used.\n")

for threshold in THRESHOLDS:
    result = evaluate_threshold(threshold)
    if result is not None:
        results.append(result)

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No valid threshold results were produced.")

print(
    f"{'Threshold':>10}"
    f"{'Tail n':>9}"
    f"{'Tail %':>10}"
    f"{'Alpha':>10}"
    f"{'95% CI for alpha':>23}"
    f"{'KS':>10}"
    f"{'GOF p*':>10}"
    f"{'Eligible':>11}"
)

for _, row in results_df.iterrows():
    ci_text = (
        f"[{row['alpha_ci_low']:.3f}, "
        f"{row['alpha_ci_high']:.3f}]"
    )

    print(
        f"{row['threshold']:>9.2%}"
        f"{int(row['n_tail']):>9}"
        f"{row['tail_fraction']:>9.2%}"
        f"{row['alpha']:>10.4f}"
        f"{ci_text:>23}"
        f"{row['ks']:>10.4f}"
        f"{row['gof_p']:>10.4f}"
        f"{str(row['eligible']):>11}"
    )


# ============================================================
# 6. PROVISIONAL THRESHOLD SELECTION
# ============================================================

eligible = results_df[results_df["eligible"]].copy()

if eligible.empty:
    raise ValueError(
        "No threshold has enough tail observations. "
        "Reconsider the candidate range using training data."
    )

# Predefined exploratory rule:
# choose the eligible threshold with the smallest two-sided KS distance.
selected = eligible.loc[eligible["ks"].idxmin()]

print("\n" + "=" * 88)
print("PROVISIONAL TRAINING-ONLY THRESHOLD CHOICE")
print("=" * 88)
print(f"Selected threshold: {selected['threshold']:.2%}")
print(f"Tail observations:  {int(selected['n_tail'])}")
print(f"Tail fraction:      {selected['tail_fraction']:.2%}")
print(f"Estimated alpha:    {selected['alpha']:.4f}")
print(f"KS distance:        {selected['ks']:.4f}")
print(
    f"Fixed-threshold bootstrap GOF p-value: "
    f"{selected['gof_p']:.4f}"
)

print("\n" + "=" * 88)
print("INTERPRETATION AND LIMITATIONS")
print("=" * 88)
print(
    "1. A smaller KS distance means a closer empirical-tail fit, "
    "not proof of a true power law.\n"
    "2. Alpha and its bootstrap interval show how the estimated "
    "tail exponent varies under the fitted model.\n"
    "3. The fixed-threshold GOF p-value does not account for "
    "searching across multiple thresholds.\n"
    "4. The selected threshold is provisional; it is not a "
    "definitive optimal threshold.\n"
    "5. These bootstrap intervals quantify sampling variation "
    "conditional on the fitted Pareto model; they do not cover "
    "all model uncertainty.\n"
    "6. This script does not model volatility clustering.\n"
    "7. No test data was used. Do not choose a model based on "
    "which setting best matches the held-out test results.\n"
)

# Save the diagnostic table for later review.
output_file = Path.cwd() / "hybrid_threshold_diagnostics.csv"
results_df.to_csv(output_file, index=False)

print(f"Diagnostic table saved to: {output_file}")
print("Validation complete.")
