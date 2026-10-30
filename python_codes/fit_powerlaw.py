
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. LOAD TRAINING DATA ONLY
# ============================================================
file_path = Path.home() / "reliance_train.csv"
data = pd.read_csv(file_path)

returns = data["Log_Return"].dropna().to_numpy()
returns = returns[np.isfinite(returns)]

if len(returns) < 100:
    raise ValueError("Unexpectedly few training returns.")

magnitudes = np.abs(returns)
N = len(returns)

# Candidate thresholds, specified before this diagnostic
thresholds = np.array(
    [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
)

rng = np.random.default_rng(42)


# ============================================================
# 2. PARETO FIT AND KS DISTANCE
# ============================================================
def fit_alpha(tail, u):
    """Continuous Pareto maximum-likelihood exponent."""
    tail = np.asarray(tail)
    if len(tail) < 2 or np.any(tail < u):
        return np.nan
    denominator = np.sum(np.log(tail / u))
    if denominator <= 0:
        return np.nan
    return len(tail) / denominator


def pareto_ks(tail, u, alpha):
    """Two-sided empirical-CDF KS distance."""
    x = np.sort(np.asarray(tail))
    n = len(x)

    if n < 2 or not np.isfinite(alpha):
        return np.nan

    fitted_cdf = 1 - (u / x) ** alpha
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(0, n) / n

    return max(
        np.max(empirical_upper - fitted_cdf),
        np.max(fitted_cdf - empirical_lower)
    )


def fit_candidates(values):
    """Fit all candidate thresholds and select the lowest KS."""
    results = []

    for u in thresholds:
        tail = values[values >= u]

        if len(tail) < 2:
            continue

        alpha = fit_alpha(tail, u)
        ks = pareto_ks(tail, u, alpha)

        results.append({
            "threshold": u,
            "n": len(tail),
            "alpha": alpha,
            "ks": ks
        })

    return results


# ============================================================
# 3. COMPARE THRESHOLDS AND SELECT THE BEST CANDIDATE
# ============================================================
results = fit_candidates(magnitudes)
best = min(results, key=lambda row: row["ks"])

u_hat = best["threshold"]
tail = magnitudes[magnitudes >= u_hat]
alpha_hat = best["alpha"]
ks_hat = best["ks"]
n_tail = len(tail)

print("=" * 72)
print("POWER-LAW TAIL DIAGNOSTICS — TRAINING DATA ONLY")
print("=" * 72)
print("Training observations:", N)
print("\nTHRESHOLD COMPARISON")
print(f"{'Threshold':>10} | {'Count':>7} | {'Alpha':>8} | {'KS':>8}")
print("-" * 44)

for row in results:
    marker = "  <-- selected" if row["threshold"] == u_hat else ""
    print(
        f"{row['threshold']:9.2%} | {row['n']:7d} | "
        f"{row['alpha']:8.4f} | {row['ks']:8.4f}{marker}"
    )

print("\nSELECTED CANDIDATE")
print("Threshold:", f"{u_hat:.2%}")
print("Tail observations:", n_tail)
print("Fraction of all observations in tail:", f"{n_tail / N:.2%}")
print("Estimated alpha:", f"{alpha_hat:.4f}")
print("Observed KS distance:", f"{ks_hat:.4f}")


# ============================================================
# 4. EXPONENT UNCERTAINTY AT THE SELECTED THRESHOLD
# ============================================================
# Parametric bootstrap, holding the threshold and tail count fixed.
B_CI = 2000
alpha_boot = np.empty(B_CI)

for b in range(B_CI):
    uniform = rng.random(n_tail)
    simulated_tail = u_hat * (1 - uniform) ** (-1 / alpha_hat)
    alpha_boot[b] = fit_alpha(simulated_tail, u_hat)

ci_low, ci_high = np.quantile(alpha_boot, [0.025, 0.975])

print("\nEXPONENT UNCERTAINTY")
print("Approximate 95% parametric bootstrap interval:",
      f"[{ci_low:.4f}, {ci_high:.4f}]")
print("Note: this interval holds the selected threshold fixed.")


# ============================================================
# 5. BOOTSTRAP THAT REPEATS THRESHOLD SELECTION
# ============================================================
# Semiparametric null:
# resample observations below selected u from the empirical body,
# simulate observations above u from the fitted Pareto tail,
# then repeat the same candidate-threshold selection procedure.
#
# This accounts for selection among our specified candidates,
# but remains an approximate goodness-of-fit procedure.

body = magnitudes[magnitudes < u_hat]
n_body = len(body)

B_GOODNESS = 1000
selected_ks_boot = np.empty(B_GOODNESS)
selected_u_boot = np.empty(B_GOODNESS)

if n_body == 0:
    raise ValueError("No body observations below selected threshold.")

for b in range(B_GOODNESS):
    simulated_body = rng.choice(body, size=n_body, replace=True)

    uniform = rng.random(n_tail)
    simulated_tail = u_hat * (1 - uniform) ** (-1 / alpha_hat)

    simulated_full = np.concatenate([simulated_body, simulated_tail])
    simulated_results = fit_candidates(simulated_full)

    chosen = min(simulated_results, key=lambda row: row["ks"])
    selected_ks_boot[b] = chosen["ks"]
    selected_u_boot[b] = chosen["threshold"]

p_value = (
    1 + np.sum(selected_ks_boot >= ks_hat)
) / (B_GOODNESS + 1)

print("\nTHRESHOLD-SELECTION-AWARE BOOTSTRAP")
print("Bootstrap repetitions:", B_GOODNESS)
print("Observed selected KS:", f"{ks_hat:.4f}")
print("Bootstrap KS 95th percentile:",
      f"{np.quantile(selected_ks_boot, 0.95):.4f}")
print("Approximate bootstrap p-value:", f"{p_value:.4f}")
print("Most frequently selected bootstrap threshold:",
      f"{thresholds[np.argmax([
          np.sum(selected_u_boot == u) for u in thresholds
      ])]:.2%}")
print("Interpretation: a small p-value is evidence against this")
print("Pareto-tail fitting procedure; a large p-value is not proof")
print("that the power law is true.")


# ============================================================
# 6. POSITIVE AND NEGATIVE TAIL DIAGNOSTICS
# ============================================================
print("\nPOSITIVE VS NEGATIVE EXTREMES")
print(f"{'Side':>10} | {'Count':>7} | {'Alpha':>8} | {'KS':>8}")
print("-" * 44)

for name, side in [
    ("Positive", returns[returns >= u_hat]),
    ("Negative", -returns[returns <= -u_hat])
]:
    if len(side) >= 2:
        side_alpha = fit_alpha(side, u_hat)
        side_ks = pareto_ks(side, u_hat, side_alpha)
        print(
            f"{name:>10} | {len(side):7d} | "
            f"{side_alpha:8.4f} | {side_ks:8.4f}"
        )

print("\nSeparate-side fits are diagnostic only; smaller samples")
print("mean more uncertainty. Different estimates alone do not")
print("prove the two tail exponents differ.")


# ============================================================
# 7. LOG-LOG PLOT: EMPIRICAL VS FITTED TAIL
# ============================================================
sorted_tail = np.sort(tail)
x_unique = np.unique(sorted_tail)

# Empirical conditional survival P(X >= x | X >= u)
survival = np.array([
    np.mean(tail >= x) for x in x_unique
])

fitted_survival = (u_hat / x_unique) ** alpha_hat

plt.figure(figsize=(9, 6))
plt.loglog(
    x_unique, survival, "o",
    markersize=4, label="Empirical tail survival"
)
plt.loglog(
    x_unique, fitted_survival, "-",
    label=f"Pareto fit (alpha={alpha_hat:.3f})"
)
plt.xlabel("Absolute daily log return")
plt.ylabel("Conditional survival probability")
plt.title("Reliance: Empirical Tail vs Power-Law Fit")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. THRESHOLD STABILITY PLOT
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(
    [r["threshold"] * 100 for r in results],
    [r["alpha"] for r in results],
    marker="o"
)
plt.xlabel("Tail threshold (%)")
plt.ylabel("Estimated power-law exponent alpha")
plt.title("Power-Law Exponent Stability Across Thresholds")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9. SUMMARY OF THE FIVE CHECKS
# ============================================================
print("\n" + "=" * 72)
print("FIVE-CHECK SUMMARY")
print("=" * 72)
print("1. Log-log straight-line region: inspect the first plot.")
print("2. Fitted exponent: see alpha and empirical-vs-fitted curve.")
print("3. Goodness of fit: see the approximate bootstrap p-value.")
print("4. Tail sample size:", n_tail, "observations at selected threshold.")
print("5. Stability: inspect the exponent-vs-threshold plot.")
print("\nNo single statistic proves a power law.")
print("All conclusions are limited to the training period and")
print("the candidate thresholds specified in this script.")
