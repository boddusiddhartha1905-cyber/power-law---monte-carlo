
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------
# POWER LAW - MONTE CARLO SIMULATION
# Download 10 years of daily data for four assets
# --------------------------------------------------

PROJECT_DIR = Path.home() / "Documents" / "power law - monte carlo"
OUTPUT_DIR = PROJECT_DIR / "asset_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "titan": "TITAN.NS",
    "jio_financial": "JIOFIN.NS",
    "gold": "GOLDBEES.NS",
    "silver": "SILVERBEES.NS",
}

# Request approximately 10 years of daily market data.
# Yahoo Finance's end date is exclusive.
END_DATE = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
START_DATE = END_DATE - pd.DateOffset(years=10)

# Same chronological 80/20 split for each asset.
TRAIN_FRACTION = 0.80


def download_asset(name, ticker):
    print(f"\nDownloading {name.upper()} ({ticker})...")

    try:
        data = yf.download(
            ticker,
            start=START_DATE.strftime("%Y-%m-%d"),
            end=END_DATE.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            timeout=30,
        )

        if data is None or data.empty:
            print(f"ERROR: No data returned for {ticker}.")
            print("Check the ticker on Yahoo Finance and try again.")
            return

        # Recent yfinance versions may return MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.loc[:, ~data.columns.duplicated()]
        data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()

        # Keep standard price and volume columns when available.
        desired = [
            "Open", "High", "Low", "Close", "Adj Close",
            "Volume", "Dividends", "Stock Splits"
        ]
        available = [col for col in desired if col in data.columns]
        data = data[available].copy()

        if "Close" not in data.columns:
            print(f"ERROR: Close price missing for {ticker}.")
            return

        # Adjusted close accounts for dividends/splits when provided.
        # Prefer it for return calculations; fall back to Close.
        price_col = (
            "Adj Close"
            if "Adj Close" in data.columns
            and data["Adj Close"].notna().any()
            else "Close"
        )

        prices = pd.to_numeric(data[price_col], errors="coerce")
        prices = prices.where(prices > 0)

        # Calculate log returns; first observation is unavailable.
        data["Log_Return"] = np.log(prices / prices.shift(1))
        data["Return_Price_Source"] = price_col

        # Save full raw data plus calculated returns.
        raw_file = OUTPUT_DIR / f"{name}_10y.csv"
        data.to_csv(raw_file, index_label="Date")

        # Drop only rows where a return cannot be calculated.
        returns = data.loc[data["Log_Return"].notna(), ["Log_Return"]].copy()
        returns.index.name = "Date"

        if len(returns) < 100:
            print(f"WARNING: Only {len(returns)} usable returns.")
            print("Check whether the ticker has enough history.")
            return

        # Chronological split; never randomly shuffle time series.
        split_idx = int(len(returns) * TRAIN_FRACTION)
        train = returns.iloc[:split_idx].copy()
        test = returns.iloc[split_idx:].copy()

        train_file = OUTPUT_DIR / f"{name}_train.csv"
        test_file = OUTPUT_DIR / f"{name}_test.csv"
        returns_file = OUTPUT_DIR / f"{name}_returns.csv"

        train.to_csv(train_file)
        test.to_csv(test_file)
        returns.to_csv(returns_file)

        print(f"Saved raw prices: {raw_file}")
        print(f"Saved returns:    {returns_file}")
        print(f"Saved training:   {train_file}")
        print(f"Saved test:       {test_file}")
        print(f"Price column:     {price_col}")
        print(f"Return count:     {len(returns)}")
        print(f"Training count:   {len(train)}")
        print(f"Test count:       {len(test)}")
        print(f"Training dates:   {train.index.min().date()} to "
              f"{train.index.max().date()}")
        print(f"Test dates:       {test.index.min().date()} to "
              f"{test.index.max().date()}")

        if len(train) and len(test):
            print(f"Training mean:    {train['Log_Return'].mean():.6f}")
            print(f"Training daily SD:{train['Log_Return'].std():.6f}")
            print(f"Test mean:        {test['Log_Return'].mean():.6f}")
            print(f"Test daily SD:    {test['Log_Return'].std():.6f}")

    except Exception as exc:
        print(f"ERROR downloading {ticker}: {exc}")


def main():
    print("POWER LAW - MONTE CARLO SIMULATION")
    print(f"Requested period: {START_DATE.date()} to "
          f"{(END_DATE - pd.Timedelta(days=1)).date()}")
    print(f"Output folder: {OUTPUT_DIR}")

    for name, ticker in ASSETS.items():
        download_asset(name, ticker)
        time.sleep(1)

    print("\nDownload process finished.")
    print("Review any ERROR messages above before using the data.")
    print(f"All output files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
