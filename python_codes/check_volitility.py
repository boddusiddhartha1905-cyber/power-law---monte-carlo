
import pandas as pd
import matplotlib.pyplot as plt

# Load TRAINING data only — do not load the test data
file_path = "/Users/siddu2007/reliance_train.csv"
df = pd.read_csv(file_path)

# Display column names so we can verify the data
print("Columns:", df.columns.tolist())

# Adjust these names if your CSV uses different column names
date_col = "Date"
return_col = "Log_Return"

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

# Convert returns to numeric values
returns = pd.to_numeric(df[return_col], errors="coerce")
df = df.loc[returns.notna()].copy()
df["Log_Return"] = returns.dropna().values

# Calculate volatility over each rolling 20-trading-day window
df["Rolling_Volatility"] = (
    df["Log_Return"].rolling(window=20).std() * 100
)

# Calculate how strongly absolute returns relate to the next day
absolute_returns = df["Log_Return"].abs()
autocorrelation = absolute_returns.autocorr(lag=1)

print("\nVOLATILITY CHECK — TRAINING DATA ONLY")
print("-------------------------------------")
print(f"Training observations: {len(df)}")
print(f"Average daily return: {df['Log_Return'].mean() * 100:.4f}%")
print(
    "Average 20-day rolling volatility: "
    f"{df['Rolling_Volatility'].mean():.4f}%"
)
print(
    "Maximum 20-day rolling volatility: "
    f"{df['Rolling_Volatility'].max():.4f}%"
)
print(
    "Minimum 20-day rolling volatility: "
    f"{df['Rolling_Volatility'].min():.4f}%"
)
print(
    "Lag-1 autocorrelation of absolute returns: "
    f"{autocorrelation:.4f}"
)

# Plot how volatility changes over time
plt.figure(figsize=(12, 5))
plt.plot(df[date_col], df["Rolling_Volatility"])
plt.title("Reliance: 20-Day Rolling Volatility (Training Period)")
plt.xlabel("Date")
plt.ylabel("Daily volatility (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_volatility.png", dpi=150)
plt.show()

