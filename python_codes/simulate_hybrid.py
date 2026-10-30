
import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# 1. Load training data and fit model parameters
# ==================================================
file_path = Path.home() / "reliance_train.csv"
data = pd.read_csv(file_path)

returns = data["Log_Return"].dropna().to_numpy()
returns = returns[np.isfinite(returns)]

u = 0.025  # Provisional 2.5% absolute log-return threshold

is_tail = np.abs(returns) >= u
body = returns[~is_tail]
tail = returns[is_tail]
tail_magnitudes = np.abs(tail)

mu_body = np.mean(body)
sigma_body = np.std(body, ddof=1)

alpha = len(tail_magnitudes) / np.sum(
    np.log(tail_magnitudes / u)
)

p_tail = len(tail) / len(returns)
p_negative = np.mean(tail < 0)

# ==================================================
# 2. Simulate a truncated-normal body
# ==================================================
def sample_truncated_normal(rng, size, mu, sigma, threshold):
    """Rejection-sample normal draws inside (-threshold, threshold)."""
    samples = np.empty(size)
    filled = 0

    while filled < size:
        needed = size - filled
        candidates = rng.normal(mu, sigma, size=max(needed * 2, 100))
        accepted = candidates[np.abs(candidates) < threshold]

        take = min(len(accepted), needed)
        samples[filled:filled + take] = accepted[:take]
        filled += take

    return samples


# ==================================================
# 3. Simulate hybrid daily returns
# ==================================================
def simulate_hybrid(size, seed=42):
    rng = np.random.default_rng(seed)

    choose_tail = rng.random(size) < p_tail
    simulated = np.empty(size)

    n_body = np.sum(~choose_tail)
    n_tail = np.sum(choose_tail)

    # Body: truncated normal
    simulated[~choose_tail] = sample_truncated_normal(
        rng, n_body, mu_body, sigma_body, u
    )

    # Tail: Pareto magnitude and observed tail direction
    uniform = rng.random(n_tail)
    uniform = np.maximum(uniform, np.finfo(float).tiny)

    magnitudes = u * (1 - uniform) ** (-1 / alpha)

    negative = rng.random(n_tail) < p_negative
    signs = np.where(negative, -1.0, 1.0)

    simulated[choose_tail] = magnitudes * signs

    return simulated


# ==================================================
# 4. Generate 100,000 synthetic daily returns
# ==================================================
N_SIM = 100_000
simulated = simulate_hybrid(N_SIM)

# ==================================================
# 5. Check the simulation
# ==================================================
simulated_tail = np.abs(simulated) >= u
simulated_body = ~simulated_tail

print("=" * 55)
print("HYBRID MONTE CARLO SIMULATION")
print("=" * 55)

print("Simulated observations:", len(simulated))
print(f"Threshold: {u:.2%}")

print("\nBODY / TAIL FREQUENCIES")
print(f"Simulated body fraction: {np.mean(simulated_body):.4%}")
print(f"Target body fraction: {1 - p_tail:.4%}")
print(f"Simulated tail fraction: {np.mean(simulated_tail):.4%}")
print(f"Target tail fraction: {p_tail:.4%}")

print("\nSIMULATED RETURN STATISTICS")
print(f"Mean log return: {np.mean(simulated):.5%}")
print(f"Daily volatility: {np.std(simulated, ddof=1):.5%}")
print(f"Largest simulated rise: {np.max(simulated):.5%}")
print(f"Largest simulated fall: {np.min(simulated):.5%}")

print("\nEXTREME MOVEMENT COUNTS")
print("Threshold | Falls | Rises")

for threshold in [0.02, 0.03, 0.05, 0.08]:
    falls = np.sum(simulated <= -threshold)
    rises = np.sum(simulated >= threshold)

    print(
        f"{threshold:7.0%} | {falls:5d} | {rises:5d}"
    )

# ==================================================
# 6. Basic consistency checks
# ==================================================
assert np.all(np.isfinite(simulated))
assert np.all(np.abs(simulated[simulated_body]) < u)
assert np.all(np.abs(simulated[simulated_tail]) >= u)

print("\nAll basic simulation checks passed.")
print("This is a synthetic sample, not a forecast.")
