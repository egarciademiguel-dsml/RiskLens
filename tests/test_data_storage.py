"""Tests for the yfinance cache layer."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.data import storage
from src.data.storage import get_or_fetch, clear_cache, list_cached


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": np.random.rand(5),
            "High": np.random.rand(5),
            "Low": np.random.rand(5),
            "Close": np.random.rand(5),
            "Volume": np.random.randint(100, 1000, 5),
        },
        index=pd.Index(dates, name="Date"),
    )


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(storage, "DATA_DIR", d)
    yield d


class TestGetOrFetch:
    def test_miss_calls_fetch_and_caches(self, sample_df, tmp_data_dir):
        with patch("src.data.storage.fetch_asset_data", return_value=sample_df) as mock:
            df = get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            assert mock.call_count == 1
            assert len(df) == len(sample_df)
            assert (tmp_data_dir / "BTC-USD_2024-01-01_2024-01-05.csv").exists()

    def test_hit_skips_fetch(self, sample_df, tmp_data_dir):
        with patch("src.data.storage.fetch_asset_data", return_value=sample_df) as mock:
            get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            assert mock.call_count == 1  # second call hit the cache

    def test_different_date_range_is_different_key(self, sample_df, tmp_data_dir):
        with patch("src.data.storage.fetch_asset_data", return_value=sample_df) as mock:
            get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            get_or_fetch("BTC-USD", start_date="2024-02-01", end_date="2024-02-05")
            assert mock.call_count == 2


class TestCacheAdmin:
    def test_clear_cache(self, sample_df, tmp_data_dir):
        with patch("src.data.storage.fetch_asset_data", return_value=sample_df):
            get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            get_or_fetch("AAPL", start_date="2024-01-01", end_date="2024-01-05")
        assert clear_cache() == 2
        assert list(tmp_data_dir.glob("*.csv")) == []

    def test_clear_cache_empty(self, tmp_data_dir):
        assert clear_cache() == 0

    def test_list_cached(self, sample_df, tmp_data_dir):
        with patch("src.data.storage.fetch_asset_data", return_value=sample_df):
            get_or_fetch("BTC-USD", start_date="2024-01-01", end_date="2024-01-05")
            get_or_fetch("AAPL", start_date="2024-01-01", end_date="2024-01-05")
        assert set(list_cached()) == {
            "BTC-USD_2024-01-01_2024-01-05",
            "AAPL_2024-01-01_2024-01-05",
        }
