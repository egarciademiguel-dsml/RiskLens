from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

from src.config import DEFAULT_PERIOD_YEARS


def fetch_asset_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance.

    Args:
        ticker: Asset symbol (e.g. "BTC-USD", "AAPL").
        start_date: ISO date string. Defaults to 5 years ago.
        end_date: ISO date string. Defaults to today.

    Returns:
        Raw OHLCV dataframe with a datetime index.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (
            datetime.today() - timedelta(days=DEFAULT_PERIOD_YEARS * 365)
        ).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Flatten multi-level columns if present (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df
