
from pathlib import Path
import numpy as np
import pandas as pd

BASE = (
    Path.home()
    / "Documents"
    / "power law - monte carlo"
    / "asset_data"
)

for asset in ["titan", "jio_financial"]:
    path = BASE / f"{asset}_10y.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")

    price_col = (
        "Adj Close"
        if "Adj Close" in df.columns
        and df["Adj Close"].notna().any()
        else "Close"
    )

    price = pd.to_numeric(df[price_col], errors="coerce")
    returns = np.log(price / price.shift(1))

    print("\n" + "=" * 60)
    print(asset.upper())
    print("=" * 60)
    print("Rows:", len(df))
    print("Price source:", price_col)
    print("Daily return SD:", round(returns.std(), 6))
    print("1% return quantile:", round(returns.quantile(0.01), 6))
    print("99% return quantile:", round(returns.quantile(0.99), 6))

    print("\nLargest absolute returns:")
    idx = returns.abs().nlargest(10).index
    report = pd.DataFrame({
        "Date": df.loc[idx, "Date"].values,
        "Close": df.loc[idx, "Close"].values,
        "Return": returns.loc[idx].values,
        "Volume": df.loc[idx, "Volume"].values,
    })
    print(report.sort_values(
        "Return", key=abs, ascending=False
    ).to_string(index=False))

    print("\nMissing close prices:", int(df["Close"].isna().sum()))
    print("Non-positive close prices:",
          int((df["Close"] <= 0).sum()))

