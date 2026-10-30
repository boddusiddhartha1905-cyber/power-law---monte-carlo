
from pathlib import Path
import pandas as pd

BASE = (
    Path.home()
    / "Documents"
    / "power law - monte carlo"
    / "asset_data"
)

checks = {
    "gold": ["2019-12-16", "2019-12-30"],
    "silver": ["2026-01-19", "2026-02-10"],
    "titan": ["2017-11-01", "2017-11-10"],
    "jio_financial": ["2024-02-01", "2024-02-12"],
}

for asset, (start, end) in checks.items():
    path = BASE / f"{asset}_10y.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    print("\n" + "=" * 60)
    print(asset.upper(), start, "to", end)
    print("=" * 60)

    cols = [
        c for c in
        ["Date", "Open", "High", "Low", "Close", "Adj Close",
         "Volume", "Stock Splits", "Dividends"]
        if c in df.columns
    ]

    window = df.loc[
        df["Date"].between(start, end), cols
    ]
    print(window.to_string(index=False))

