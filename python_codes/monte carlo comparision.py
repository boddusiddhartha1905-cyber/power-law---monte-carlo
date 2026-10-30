
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# POWER-LAW MONTE CARLO SIMULATION
# Models: Normal benchmark vs Power-law hybrid
# Fit on training data only; evaluate on the final two years.
# ============================================================

PROJECT = Path("/Users/siddu2007/Documents/power law - monte carlo")
TRAIN_FILE = Path("/Users/siddu2007/reliance_train.csv")
TEST_FILE = Path("/Users/siddu2007/reliance_test.csv")

OUTPUT = PROJECT / "monte_carlo_results"
OUTPUT.mkdir(exist_ok=True)

N_PATHS = 10_000
SEED = 42
TAIL_THRESHOLD = 0.025  # provisional 2.5% threshold
rng = np.random.default_rng(SEED)


def load_returns(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Log_Return"] = pd.to_numeric(df["Log_Return"], errors="coerce")
    df = df.dropna(subset=["Log_Return"]).reset_index(drop=True)
    return df


train = load_returns(TRAIN_FILE)
test = load_returns(TEST_FILE)

train_r = train["Log_Return"].to_numpy(dtype=float)
test_r = test["Log_Return"].to_numpy(dtype=float)

if len(train_r) == 0 or len(test_r) == 0:
    raise ValueError("Training or test returns are empty.")

if not np.all(np.isfinite(train_r)) or not np.all(np.isfinite(test_r)):
    raise ValueError("Returns contain non-finite values.")

if np.any(np.abs(train_r) >= TAIL_THRESHOLD):
    # Expected: tail observations exist; this check is informational.
    pass

# IMPORTANT: test returns are used only for evaluation.
# All model parameters below are estimated from training returns.


# -------------------------
# Model 1: Normal benchmark
# -------------------------
normal_mu = train_r.mean()
normal_sigma = train_r.std(ddof=1)


# -------------------------------------------
# Model 2: Normal body + Pareto power-law tail
# -------------------------------------------
body = train_r[np.abs(train_r) < TAIL_THRESHOLD]
tail_returns = train_r[np.abs(train_r) >= TAIL_THRESHOLD]
tail_magnitudes = np.abs(tail_returns)

if len(body) == 0 or len(tail_magnitudes) < 50:
    raise ValueError("Insufficient body or tail observations.")

tail_alpha = len(tail_magnitudes) / np.sum(
    np.log(tail_magnitudes / TAIL_THRESHOLD)
)

tail_probability = len(tail_returns) / len(train_r)
negative_tail_probability = np.mean(tail_returns < 0)

# Empirical body parameters, estimated from training data only.
body_mu = body.mean()
body_sigma = body.std(ddof=1)

if body_sigma <= 0 or tail_alpha <= 0:
    raise ValueError("Invalid fitted model parameters.")


def sample_truncated_normal(n, mu, sigma, limit, random_state):
    """Sample normal values conditional on absolute value < limit."""
    result = np.empty(n)
    filled = 0

    while filled < n:
        batch_size = max(1000, 2 * (n - filled))
        candidates = random_state.normal(mu, sigma, batch_size)
        candidates = candidates[np.abs(candidates) < limit]

        take = min(len(candidates), n - filled)
        if take:
            result[filled:filled + take] = candidates[:take]
            filled += take

    return result


def simulate_normal(n_paths, horizon, random_state):
    return random_state.normal(
        normal_mu, normal_sigma, size=(n_paths, horizon)
    )


def simulate_powerlaw_hybrid(n_paths, horizon, random_state):
    """Simulate independent daily returns from the fitted hybrid."""
    n_total = n_paths * horizon

    is_tail = random_state.random(n_total) < tail_probability
    simulated = np.empty(n_total)

    n_body = int((~is_tail).sum())
    simulated[~is_tail] = sample_truncated_normal(
        n_body, body_mu, body_sigma, TAIL_THRESHOLD, random_state
    )

    n_tail = int(is_tail.sum())
    if n_tail:
        # Pareto survival: P(X >= x) = (threshold/x)^alpha
        magnitudes = TAIL_THRESHOLD * (
            1 - random_state.random(n_tail)
        ) ** (-1 / tail_alpha)

        is_negative = (
            random_state.random(n_tail) < negative_tail_probability
        )
        simulated[is_tail] = np.where(is_negative, -magnitudes, magnitudes)

    return simulated.reshape(n_paths, horizon)


# ------------------------------------
# Simulate the full two-year horizon
# ------------------------------------
horizon = len(test_r)

normal_paths = simulate_normal(N_PATHS, horizon, rng)
hybrid_paths = simulate_powerlaw_hybrid(N_PATHS, horizon, rng)

# Build price paths from the final training close.
if "Close" not in train.columns or "Close" not in test.columns:
    raise ValueError("Both CSV files must contain a Close column.")

start_price = float(train["Close"].iloc[-1])
actual_prices = start_price * np.exp(np.cumsum(test_r))

normal_prices = start_price * np.exp(np.cumsum(normal_paths, axis=1))
hybrid_prices = start_price * np.exp(np.cumsum(hybrid_paths, axis=1))


# -------------------------
# Evaluation helper metrics
# -------------------------
def var_es(returns, confidence=0.95):
    losses = -np.asarray(returns)
    var = np.quantile(losses, confidence)
    tail_losses = losses[losses >= var]
    es = tail_losses.mean() if len(tail_losses) else var
    return var, es


def path_metrics(return_paths, price_paths):
    final_prices = price_paths[:, -1]
    peak_prices = np.maximum.accumulate(
        np.column_stack([
            np.full(len(price_paths), start_price), price_paths
        ]),
        axis=1
    )
    price_with_start = np.column_stack([
        np.full(len(price_paths), start_price), price_paths
    ])
    drawdowns = price_with_start / peak_prices - 1
    max_drawdowns = drawdowns.min(axis=1)

    return {
        "Final price median": np.median(final_prices),
        "Final price 5th percentile": np.quantile(final_prices, 0.05),
        "Final price 95th percentile": np.quantile(final_prices, 0.95),
        "Probability final price below start": np.mean(final_prices < start_price),
        "Median maximum drawdown": np.median(max_drawdowns),
        "5th percentile maximum drawdown": np.quantile(max_drawdowns, 0.05),
        "Median realised volatility": np.median(
            np.std(return_paths, axis=1, ddof=1) * np.sqrt(252)
        ),
    }


actual_var, actual_es = var_es(test_r)
normal_var, normal_es = var_es(normal_paths.ravel())
hybrid_var, hybrid_es = var_es(hybrid_paths.ravel())

summary = pd.DataFrame([
    {
        "Model": "Actual test period",
        "Mean daily return": test_r.mean(),
        "Daily volatility": test_r.std(ddof=1),
        "VaR 95% loss": actual_var,
        "Expected shortfall 95%": actual_es,
        **{
            "Final price median": actual_prices[-1],
            "Final price 5th percentile": np.nan,
            "Final price 95th percentile": np.nan,
            "Probability final price below start": float(actual_prices[-1] < start_price),
            "Median maximum drawdown": np.nan,
            "5th percentile maximum drawdown": np.nan,
            "Median realised volatility": np.std(test_r, ddof=1) * np.sqrt(252),
        },
    },
    {
        "Model": "Normal",
        "Mean daily return": normal_paths.mean(),
        "Daily volatility": normal_paths.std(ddof=1),
        "VaR 95% loss": normal_var,
        "Expected shortfall 95%": normal_es,
        **path_metrics(normal_paths, normal_prices),
    },
    {
        "Model": "Power-law hybrid",
        "Mean daily return": hybrid_paths.mean(),
        "Daily volatility": hybrid_paths.std(ddof=1),
        "VaR 95% loss": hybrid_var,
        "Expected shortfall 95%": hybrid_es,
        **path_metrics(hybrid_paths, hybrid_prices),
    },
])

summary.to_csv(OUTPUT / "model_summary.csv", index=False)

# Daily return distribution comparison.
daily_counts = []
thresholds = [0.02, 0.03, 0.05, 0.08]

for threshold in thresholds:
    daily_counts.append({
        "Absolute return threshold": threshold,
        "Actual negative count": int(np.sum(test_r <= -threshold)),
        "Actual positive count": int(np.sum(test_r >= threshold)),
        "Normal expected negative count": float(
            N_PATHS * np.mean(normal_paths <= -threshold) / N_PATHS
            * horizon
        ),
        "Normal expected positive count": float(
            N_PATHS * np.mean(normal_paths >= threshold) / N_PATHS
            * horizon
        ),
        "Hybrid expected negative count": float(
            N_PATHS * np.mean(hybrid_paths <= -threshold) / N_PATHS
            * horizon
        ),
        "Hybrid expected positive count": float(
            N_PATHS * np.mean(hybrid_paths >= threshold) / N_PATHS
            * horizon
        ),
    })

pd.DataFrame(daily_counts).to_csv(
    OUTPUT / "extreme_event_counts.csv", index=False
)

# Plot actual price against simulated median and 90% interval.
dates = test["Date"].to_numpy()
day_numbers = np.arange(horizon)

plt.figure(figsize=(12, 6))
plt.plot(dates, actual_prices, label="Actual test price", linewidth=2)

for prices, label in [
    (normal_prices, "Normal"),
    (hybrid_prices, "Power-law hybrid"),
]:
    median = np.median(prices, axis=0)
    lower = np.quantile(prices, 0.05, axis=0)
    upper = np.quantile(prices, 0.95, axis=0)

    plt.plot(dates, median, label=f"{label} median")
    plt.fill_between(dates, lower, upper, alpha=0.15)

plt.title("Two-Year Monte Carlo Scenario Comparison")
plt.xlabel("Date")
plt.ylabel("Simulated price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "price_scenarios.png", dpi=150)
plt.close()

print("\nTRAINING-ONLY FIT")
print("-----------------")
print(f"Training observations: {len(train_r)}")
print(f"Test horizon (trading days): {horizon}")
print(f"Normal mean: {normal_mu:.6f}")
print(f"Normal standard deviation: {normal_sigma:.6f}")
print(f"Hybrid threshold: {TAIL_THRESHOLD:.2%} (provisional)")
print(f"Hybrid tail observations: {len(tail_returns)}")
print(f"Hybrid tail probability: {tail_probability:.2%}")
print(f"Hybrid Pareto alpha: {tail_alpha:.4f}")
print(f"Hybrid body mean: {body_mu:.6f}")
print(f"Hybrid body standard deviation: {body_sigma:.6f}")

print("\nMODEL SUMMARY")
print(summary.to_string(index=False))

print("\nSaved outputs to:", OUTPUT)
print(" - model_summary.csv")
print(" - extreme_event_counts.csv")
print(" - price_scenarios.png")

