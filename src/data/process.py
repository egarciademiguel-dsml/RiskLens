import numpy as np
import pandas as pd


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, sort by date, handle missing values."""
    df = df.copy()

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure index is datetime and named 'date'
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Sort chronologically
    df = df.sort_index()

    # Keep only OHLCV columns
    expected = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in expected if c in df.columns]]

    # Forward-fill then drop any remaining NaN rows
    df = df.ffill().dropna()

    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log returns based on the close price."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Drop the first row (NaN from pct_change)
    df = df.iloc[1:]

    return df
