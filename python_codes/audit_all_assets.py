
from pathlib import Path
import numpy as np
import pandas as pd

BASE = (
    Path.home()
    / "Documents"
    / "power law - monte carlo"
    / "asset_data"
)

ASSETS = ["titan", "jio_financial", "gold", "silver"]

for asset in ASSETS:
    print("\n" + "=" * 65)
    print(f"AUDIT: {asset.upper()}")
    print("=" * 65)

    file = BASE / f"{asset}_10y.csv"
    if not file.exists():
        print("Missing file:", file)
        continue

    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")

    price_col = (
        "Adj Close"
        if "Adj Close" in df.columns
        and df["Adj Close"].notna().any()
        else "Close"
    )

    prices = pd.to_numeric(df[price_col], errors="coerce")
    returns = np.log(prices / prices.shift(1))

    print("Price source:", price_col)
    print("Rows:", len(df))
    print("Dates:", df["Date"].min(), "to", df["Date"].max())
    print("Missing prices:", int(prices.isna().sum()))
    print("Non-positive prices:", int((prices <= 0).sum()))
    print("Duplicate dates:", int(df["Date"].duplicated().sum()))

    print("\nReturn statistics:")
    print(returns.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]))

    print("\nLargest absolute daily log returns:")
    largest = returns.abs().nlargest(10).index
    report = pd.DataFrame({
        "Date": df.loc[largest, "Date"].values,
        "Price": prices.loc[largest].values,
        "Log_Return": returns.loc[largest].values,
        "Return_Percent_Approx": 100 * returns.loc[largest].values,
    })
    print(report.sort_values("Log_Return", key=abs, ascending=False)
          .to_string(index=False))

    # Compare adjusted and unadjusted close returns where possible.
    if "Close" in df.columns and "Adj Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
        close_ret = np.log(close / close.shift(1))
        adj_ret = np.log(adj / adj.shift(1))

        print("\nVolatility comparison:")
        print("Close-return daily SD:", round(close_ret.std(), 6))
        print("Adjusted-return daily SD:", round(adj_ret.std(), 6))

        differences = (adj_ret - close_ret).abs()
        idx = differences.nlargest(5).index
        print("\nLargest adjusted-vs-close return differences:")
        print(pd.DataFrame({
            "Date": df.loc[idx, "Date"].values,
            "Close": close.loc[idx].values,
            "Adj_Close": adj.loc[idx].values,
            "Close_Return": close_ret.loc[idx].values,
            "Adj_Return": adj_ret.loc[idx].values,
        }).to_string(index=False))
