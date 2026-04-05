import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.data.process import clean_market_data, add_returns
from src.data.validate import validate_ticker


@pytest.fixture
def raw_ohlcv():
    """Simulates raw yfinance output."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.normal(0, 1, 100))
    return pd.DataFrame(
        {
            "Open": close + np.random.uniform(-1, 1, 100),
            "High": close + np.abs(np.random.normal(0, 1, 100)),
            "Low": close - np.abs(np.random.normal(0, 1, 100)),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


class TestCleanMarketData:

    def test_columns_lowercase(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_sorted_by_date(self, raw_ohlcv):
        shuffled = raw_ohlcv.sample(frac=1)
        df = clean_market_data(shuffled)
        assert df.index.is_monotonic_increasing

    def test_index_is_datetime_named_date(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        assert df.index.name == "date"

    def test_no_nan_after_ffill(self, raw_ohlcv):
        raw_ohlcv.iloc[5, 0] = np.nan
        raw_ohlcv.iloc[10, 3] = np.nan
        df = clean_market_data(raw_ohlcv)
        assert not df.isna().any().any()

    def test_drops_extra_columns(self, raw_ohlcv):
        raw_ohlcv["Extra"] = 999
        raw_ohlcv["Adj Close"] = raw_ohlcv["Close"]
        df = clean_market_data(raw_ohlcv)
        assert "extra" not in df.columns
        assert "adj close" not in df.columns

    def test_preserves_row_count(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        assert len(df) == len(raw_ohlcv)


class TestAddReturns:

    def test_adds_returns_columns(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        df = add_returns(df)
        assert "returns" in df.columns
        assert "log_returns" in df.columns

    def test_drops_first_row(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        len_before = len(df)
        df = add_returns(df)
        assert len(df) == len_before - 1

    def test_no_nan_in_returns(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        df = add_returns(df)
        assert not df["returns"].isna().any()
        assert not df["log_returns"].isna().any()

    def test_returns_values_are_correct(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        df = add_returns(df)
        # Manually compute expected return for second row
        expected = (df["close"].iloc[0] - clean_market_data(raw_ohlcv)["close"].iloc[0]) / clean_market_data(raw_ohlcv)["close"].iloc[0]
        assert abs(df["returns"].iloc[0] - expected) < 1e-10

    def test_monotonic_index_preserved(self, raw_ohlcv):
        df = clean_market_data(raw_ohlcv)
        df = add_returns(df)
        assert df.index.is_monotonic_increasing


class TestValidateTicker:

    def test_valid_ticker_returns_true(self):
        with patch("src.data.validate.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame({"Close": [1, 2, 3]})
            assert validate_ticker("AAPL") is True

    def test_invalid_ticker_returns_false(self):
        with patch("src.data.validate.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            assert validate_ticker("XYZXYZXYZ") is False

    def test_empty_string_returns_false(self):
        assert validate_ticker("") is False

    def test_whitespace_only_returns_false(self):
        assert validate_ticker("   ") is False
