import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from pathlib import Path

from src.data.storage import (
    save_asset_data,
    load_asset_data,
    has_cached_data,
    delete_asset_data,
    list_cached_assets,
    cleanup_all_cached_data,
    get_cache_size_bytes,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "open": np.random.rand(5),
            "high": np.random.rand(5),
            "low": np.random.rand(5),
            "close": np.random.rand(5),
            "volume": np.random.randint(100, 1000, 5),
            "returns": np.random.rand(5),
            "log_returns": np.random.rand(5),
        },
        index=pd.Index(dates, name="date"),
    )


@pytest.fixture
def tmp_processed_dir(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    with patch("src.data.storage.DATA_PROCESSED_DIR", d):
        yield d


class TestOverwritePolicy:
    """Core policy: one file per asset, overwrite on refresh."""

    def test_save_creates_file(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            path = save_asset_data(sample_df, "BTC-USD")
            assert path.exists()
            assert path.name == "BTC-USD_daily.csv"

    def test_save_overwrites_existing(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "BTC-USD")
            first_size = (tmp_processed_dir / "BTC-USD_daily.csv").stat().st_size

            # Save again with different data — must overwrite, not append
            small_df = sample_df.iloc[:2]
            save_asset_data(small_df, "BTC-USD")
            second_size = (tmp_processed_dir / "BTC-USD_daily.csv").stat().st_size

            assert second_size < first_size
            files = list(tmp_processed_dir.glob("BTC*"))
            assert len(files) == 1  # still exactly one file

    def test_multiple_assets_one_file_each(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "BTC-USD")
            save_asset_data(sample_df, "AAPL")
            files = list(tmp_processed_dir.glob("*_daily.csv"))
            assert len(files) == 2


class TestCacheOperations:

    def test_has_cached_data(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            assert not has_cached_data("BTC-USD")
            save_asset_data(sample_df, "BTC-USD")
            assert has_cached_data("BTC-USD")

    def test_load_returns_none_when_missing(self, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            assert load_asset_data("FAKE") is None

    def test_load_returns_dataframe(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "SPY")
            loaded = load_asset_data("SPY")
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == len(sample_df)

    def test_delete_asset(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "BTC-USD")
            assert delete_asset_data("BTC-USD") is True
            assert not has_cached_data("BTC-USD")

    def test_delete_nonexistent(self, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            assert delete_asset_data("FAKE") is False

    def test_list_cached_assets(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "BTC-USD")
            save_asset_data(sample_df, "AAPL")
            assets = list_cached_assets()
            assert set(assets) == {"BTC-USD", "AAPL"}


class TestCleanup:

    def test_cleanup_all(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            save_asset_data(sample_df, "BTC-USD")
            save_asset_data(sample_df, "AAPL")
            count = cleanup_all_cached_data()
            assert count == 2
            assert list_cached_assets() == []

    def test_cleanup_empty_dir(self, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            assert cleanup_all_cached_data() == 0

    def test_cache_size(self, sample_df, tmp_processed_dir):
        with patch("src.data.storage.DATA_PROCESSED_DIR", tmp_processed_dir):
            assert get_cache_size_bytes() == 0
            save_asset_data(sample_df, "BTC-USD")
            assert get_cache_size_bytes() > 0
