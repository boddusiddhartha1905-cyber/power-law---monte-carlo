
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = Path.home()
THRESHOLD = 0.025
EXTREME_LEVELS = [0.02, 0.03, 0.05, 0.08]

SIMULATIONS = 500_000
SEED = 42


# Standard normal CDF, using Python's built-in math library.
def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


# ============================================================
# 2. LOAD TRAINING DATA AND FIT BOTH MODELS
# ============================================================

train_df = pd.read_csv(DATA_DIR / "reliance_train.csv")
train = train_df["Log_Return"].dropna().to_numpy()

if len(train) == 0:
    raise ValueError("Training data contains no valid returns.")

if not np.all(np.isfinite(train)):
    raise ValueError("Training data contains invalid return values.")

# Model A: Normal distribution fitted to all training returns.
normal_mu = np.mean(train)
normal_sigma = np.std(train, ddof=1)

# Model B: Hybrid normal-body + Pareto-tail model.
is_tail = np.abs(train) >= THRESHOLD
body = train[~is_tail]
tail = train[is_tail]

if len(body) == 0 or len(tail) == 0:
    raise ValueError("Training data must contain both body and tail observations.")

body_mu = np.mean(body)
body_sigma = np.std(body, ddof=1)

tail_magnitudes = np.abs(tail)
alpha = len(tail) / np.sum(np.log(tail_magnitudes / THRESHOLD))

p_tail = len(tail) / len(train)
p_body = 1 - p_tail

p_negative = np.mean(tail < 0)
p_positive = np.mean(tail > 0)

if normal_sigma <= 0 or body_sigma <= 0 or alpha <= 0:
    raise ValueError("Invalid fitted model parameters.")


# ============================================================
# 3. LOAD THE HELD-OUT TEST DATA
# ============================================================

test_df = pd.read_csv(DATA_DIR / "reliance_test.csv")
test = test_df["Log_Return"].dropna().to_numpy()

if len(test) == 0:
    raise ValueError("Test data contains no valid returns.")

if not np.all(np.isfinite(test)):
    raise ValueError("Test data contains invalid return values.")


# ============================================================
# 4. ANALYTICAL EXTREME-MOVE PROBABILITIES
# ============================================================

def normal_probabilities(level):
    # P(return <= -level) and P(return >= level)
    p_fall = normal_cdf(-level, normal_mu, normal_sigma)
    p_rise = 1 - normal_cdf(level, normal_mu, normal_sigma)
    return p_fall, p_rise


def hybrid_probabilities(level):
    if level >= THRESHOLD:
        # Only the Pareto tail contributes.
        tail_survival = (THRESHOLD / level) ** alpha
        p_fall = p_tail * p_negative * tail_survival
        p_rise = p_tail * p_positive * tail_survival

    else:
        # Below the threshold, include:
        # 1. Body probabilities from a truncated normal.
        # 2. All tail events, because tail magnitudes >= threshold.

        f_upper = normal_cdf(THRESHOLD, body_mu, body_sigma)
        f_lower = normal_cdf(-THRESHOLD, body_mu, body_sigma)
        body_normalizer = f_upper - f_lower

        if body_normalizer <= 0:
            raise ValueError("Invalid truncated-normal normalization.")

        body_fall_conditional = (
            normal_cdf(-level, body_mu, body_sigma) - f_lower
        ) / body_normalizer

        body_rise_conditional = (
            f_upper - normal_cdf(level, body_mu, body_sigma)
        ) / body_normalizer

        p_fall = (
            p_body * body_fall_conditional
            + p_tail * p_negative
        )
        p_rise = (
            p_body * body_rise_conditional
            + p_tail * p_positive
        )

    return p_fall, p_rise


# ============================================================
# 5. SIMULATE BOTH MODELS
# ============================================================

rng = np.random.default_rng(SEED)

normal_sim = rng.normal(
    loc=normal_mu,
    scale=normal_sigma,
    size=SIMULATIONS,
)

# Hybrid model: sample body or tail according to training frequency.
is_sim_tail = rng.random(SIMULATIONS) < p_tail
hybrid_sim = np.empty(SIMULATIONS)

n_tail_sim = np.sum(is_sim_tail)
n_body_sim = SIMULATIONS - n_tail_sim

# Body draws: normal distribution conditioned on |return| < threshold.
body_draws = []
while sum(len(x) for x in body_draws) < n_body_sim:
    needed = n_body_sim - sum(len(x) for x in body_draws)
    candidates = rng.normal(body_mu, body_sigma, size=max(needed * 2, 100))
    accepted = candidates[np.abs(candidates) < THRESHOLD]
    body_draws.append(accepted[:needed])

hybrid_sim[~is_sim_tail] = np.concatenate(body_draws)[:n_body_sim]

# Tail draws: Pareto magnitudes and training-estimated sign probabilities.
uniform_draws = rng.random(n_tail_sim)
magnitudes = THRESHOLD * (1 - uniform_draws) ** (-1 / alpha)

negative_sign = rng.random(n_tail_sim) < p_negative
tail_draws = np.where(negative_sign, -magnitudes, magnitudes)
hybrid_sim[is_sim_tail] = tail_draws


# ============================================================
# 6. COMPARE EXTREME MOVES ON THE TEST SET
# ============================================================

print("\n" + "=" * 72)
print("HELD-OUT TEST: NORMAL MODEL VS HYBRID POWER-LAW MODEL")
print("=" * 72)

print(f"Training observations: {len(train)}")
print(f"Test observations:     {len(test)}")
print(f"Tail threshold:        {THRESHOLD:.2%}")
print(f"Power-law exponent:    {alpha:.4f}")

print("\nEXTREME-MOVE COUNTS")
print("Expected counts are probabilities multiplied by test sample size.")
print("Expected counts can be fractional; actual counts are integers.\n")

print(
    f"{'Move':<10}"
    f"{'Direction':<10}"
    f"{'Actual':>9}"
    f"{'Normal exp.':>14}"
    f"{'Hybrid exp.':>14}"
)

for level in EXTREME_LEVELS:
    actual_falls = np.sum(test <= -level)
    actual_rises = np.sum(test >= level)

    normal_fall_p, normal_rise_p = normal_probabilities(level)
    hybrid_fall_p, hybrid_rise_p = hybrid_probabilities(level)

    rows = [
        (
            "Fall",
            actual_falls,
            normal_fall_p * len(test),
            hybrid_fall_p * len(test),
        ),
        (
            "Rise",
            actual_rises,
            normal_rise_p * len(test),
            hybrid_rise_p * len(test),
        ),
    ]

    for direction, actual, normal_expected, hybrid_expected in rows:
        print(
            f"{level:.0%} {direction:<5}"
            f"{'':<3}"
            f"{actual:>9d}"
            f"{normal_expected:>14.2f}"
            f"{hybrid_expected:>14.2f}"
        )


# ============================================================
# 7. DOWNSIDE RISK: 95% VaR AND EXPECTED SHORTFALL
# ============================================================

def risk_metrics(returns):
    # Define losses as negative returns.
    losses = -returns

    var_95 = np.quantile(losses, 0.95)
    tail_losses = losses[losses >= var_95]
    es_95 = np.mean(tail_losses)

    return var_95, es_95


actual_var, actual_es = risk_metrics(test)
normal_var, normal_es = risk_metrics(normal_sim)
hybrid_var, hybrid_es = risk_metrics(hybrid_sim)

print("\n" + "=" * 72)
print("DOWNSIDE RISK — 95% VaR AND EXPECTED SHORTFALL")
print("=" * 72)
print("Positive values represent losses.\n")

print(f"{'Dataset / Model':<22}{'VaR 95%':>14}{'ES 95%':>14}")
print(f"{'Actual test':<22}{actual_var:>13.2%}{actual_es:>14.2%}")
print(f"{'Normal model':<22}{normal_var:>13.2%}{normal_es:>14.2%}")
print(f"{'Hybrid model':<22}{hybrid_var:>13.2%}{hybrid_es:>14.2%}")


# ============================================================
# 8. SIMPLE INTERPRETATION GUIDE
# ============================================================

print("\n" + "=" * 72)
print("HOW TO READ THE RESULTS")
print("=" * 72)
print(
    "1. Compare expected counts with actual counts at each threshold.\n"
    "2. A closer count is encouraging, but does not prove overall superiority.\n"
    "3. Compare VaR and ES to assess downside-loss estimates.\n"
    "4. The test set is small, so differences may reflect sampling noise.\n"
    "5. Both models assume independent daily returns and do not model\n"
    "   volatility clustering.\n"
    "6. The hybrid body parameters are preliminary estimates, not an exact\n"
    "   maximum-likelihood fit of a truncated normal distribution.\n"
)

print("Evaluation complete. No model parameters were fitted to test data.")
