
import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# 1. Load training data only
# ==================================================
file_path = Path.home() / "reliance_train.csv"
data = pd.read_csv(file_path)

returns = data["Log_Return"].dropna().to_numpy()
returns = returns[np.isfinite(returns)]

# Provisional threshold from our tail investigation
u = 0.025

# ==================================================
# 2. Separate body and tail
# ==================================================
is_tail = np.abs(returns) >= u

body = returns[~is_tail]
tail = returns[is_tail]
tail_magnitudes = np.abs(tail)

if len(body) < 2 or len(tail_magnitudes) < 2:
    raise ValueError("Not enough body or tail observations.")

# ==================================================
# 3. Fit the normal body
# ==================================================
mu_body = np.mean(body)
sigma_body = np.std(body, ddof=1)

if sigma_body <= 0:
    raise ValueError("Body volatility must be positive.")

# ==================================================
# 4. Fit the Pareto tail
# ==================================================
alpha = len(tail_magnitudes) / np.sum(
    np.log(tail_magnitudes / u)
)

if alpha <= 0:
    raise ValueError("Estimated tail exponent must be positive.")

# ==================================================
# 5. Estimate mixture probabilities
# ==================================================
p_tail = len(tail) / len(returns)
p_body = 1 - p_tail

p_negative = np.mean(tail < 0)
p_positive = np.mean(tail > 0)

# ==================================================
# 6. Display model parameters
# ==================================================
print("=" * 55)
print("HYBRID NORMAL + POWER-LAW MODEL")
print("=" * 55)

print("Training observations:", len(returns))
print(f"Tail threshold: {u:.2%}")

print("\nBODY: TRUNCATED NORMAL")
print("Body observations:", len(body))
print(f"Body mean: {mu_body:.8f}")
print(f"Body standard deviation: {sigma_body:.8f}")

print("\nTAIL: PARETO")
print("Tail observations:", len(tail))
print(f"Tail exponent alpha: {alpha:.4f}")

print("\nMIXTURE PROBABILITIES")
print(f"Body probability: {p_body:.4%}")
print(f"Tail probability: {p_tail:.4%}")
print(f"Negative fraction among extremes: {p_negative:.4%}")
print(f"Positive fraction among extremes: {p_positive:.4%}")

print("\nProbability check:",
      f"{p_body + p_tail:.10f}")

print("\nParameter fitting complete.")
print("No test data was loaded or used.")

