"""Data storage with no-accumulation policy.

Strategy: overwrite-on-refresh, one file per asset max.
VaRify is an analysis app, not a data warehouse.
"""

import os
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR


def save_asset_data(df: pd.DataFrame, ticker: str) -> Path:
    """Save processed data for a single asset, overwriting any previous version."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED_DIR / f"{ticker.upper()}_daily.csv"
    df.to_csv(path)
    return path


def load_asset_data(ticker: str) -> pd.DataFrame | None:
    """Load cached data for an asset. Returns None if no cache exists."""
    path = DATA_PROCESSED_DIR / f"{ticker.upper()}_daily.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


def has_cached_data(ticker: str) -> bool:
    path = DATA_PROCESSED_DIR / f"{ticker.upper()}_daily.csv"
    return path.exists()


def delete_asset_data(ticker: str) -> bool:
    """Remove cached data for a specific asset."""
    path = DATA_PROCESSED_DIR / f"{ticker.upper()}_daily.csv"
    if path.exists():
        path.unlink()
        return True
    return False


def list_cached_assets() -> list[str]:
    """List all tickers that have cached data files."""
    if not DATA_PROCESSED_DIR.exists():
        return []
    return [
        f.stem.replace("_daily", "")
        for f in DATA_PROCESSED_DIR.glob("*_daily.csv")
    ]


def cleanup_all_cached_data() -> int:
    """Remove all cached data files. Returns count of files deleted."""
    if not DATA_PROCESSED_DIR.exists():
        return 0
    files = list(DATA_PROCESSED_DIR.glob("*_daily.csv"))
    for f in files:
        f.unlink()
    return len(files)


def get_cache_size_bytes() -> int:
    """Total size of all cached data files in bytes."""
    if not DATA_PROCESSED_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in DATA_PROCESSED_DIR.glob("*_daily.csv"))
