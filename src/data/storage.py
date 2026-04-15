"""Local yfinance cache.

Single flat directory keyed by (ticker, start, end). Cache is gitignored and
local-only — Streamlit Cloud's filesystem is ephemeral, so caching there is
a no-op by design. Used to avoid repeated yfinance round-trips during local
development, notebook runs, and tests.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEFAULT_PERIOD_YEARS
from src.data.fetch import fetch_asset_data


def _resolve_dates(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    """Normalize default dates so the cache key is deterministic."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (
            datetime.today() - timedelta(days=DEFAULT_PERIOD_YEARS * 365)
        ).strftime("%Y-%m-%d")
    return start_date, end_date


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    return DATA_DIR / f"{ticker.upper()}_{start_date}_{end_date}.csv"


def get_or_fetch(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return cached OHLCV data for (ticker, start, end), fetching on miss.

    Cache is a single flat directory. Miss → call `fetch_asset_data`, write CSV,
    return. Hit → read CSV, return. Default dates are resolved before keying so
    "5 years ago → today" is a stable key within a single day.
    """
    start, end = _resolve_dates(start_date, end_date)
    path = _cache_path(ticker, start, end)

    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = fetch_asset_data(ticker, start_date=start, end_date=end)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def clear_cache() -> int:
    """Remove every cached CSV. Returns number of files deleted."""
    if not DATA_DIR.exists():
        return 0
    files = list(DATA_DIR.glob("*.csv"))
    for f in files:
        f.unlink()
    return len(files)


def list_cached() -> list[str]:
    """List cache file stems (TICKER_START_END)."""
    if not DATA_DIR.exists():
        return []
    return sorted(f.stem for f in DATA_DIR.glob("*.csv"))
