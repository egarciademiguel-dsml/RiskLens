import pandas as pd
import numpy as np
import pytest

from src.analytics.risk_metrics import (
    annualized_volatility,
    rolling_volatility,
    max_drawdown,
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    gain_to_pain_ratio,
    best_worst_periods,
    return_statistics,
)


@pytest.fixture
def returns():
    """Deterministic returns series with known properties."""
    np.random.seed(0)
    vals = np.random.normal(0.001, 0.02, 500)
    return pd.Series(vals, index=pd.date_range("2020-01-01", periods=500, freq="D"), name="returns")


@pytest.fixture
def close(returns):
    """Close price series derived from returns."""
    prices = 100 * (1 + returns).cumprod()
    prices.name = "close"
    return prices


class TestAnnualizedVolatility:

    def test_exact_value(self, returns):
        assert abs(annualized_volatility(returns) - 0.3172223334) < 1e-6

    def test_zero_vol_for_constant_returns(self):
        flat = pd.Series([0.01] * 100)
        assert annualized_volatility(flat) < 1e-10

    def test_empty_series_returns_nan(self):
        result = annualized_volatility(pd.Series(dtype=float))
        assert np.isnan(result)


class TestRollingVolatility:

    def test_length_matches_input(self, returns):
        rv = rolling_volatility(returns, window=21)
        assert len(rv) == len(returns)

    def test_first_values_are_nan(self, returns):
        rv = rolling_volatility(returns, window=21)
        assert rv.iloc[:20].isna().all()
        assert rv.iloc[20:].notna().all()


class TestDrawdown:

    def test_exact_max_drawdown(self, close):
        assert abs(max_drawdown(close) - (-0.4201459780)) < 1e-6

    def test_drawdown_series_starts_at_zero(self, close):
        dd = drawdown_series(close)
        assert dd.iloc[0] == 0.0

    def test_drawdown_never_positive(self, close):
        dd = drawdown_series(close)
        assert (dd <= 0).all()

    def test_monotonically_rising_price_has_zero_drawdown(self):
        prices = pd.Series([100, 101, 102, 103, 104])
        assert max_drawdown(prices) == 0.0


class TestSharpeRatio:

    def test_exact_value(self, returns):
        assert abs(sharpe_ratio(returns) - 0.3915664589) < 1e-6

    def test_zero_std_returns_zero(self):
        flat = pd.Series([0.0] * 100)
        assert sharpe_ratio(flat) == 0.0

    def test_negative_returns_negative_sharpe(self):
        r = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])
        assert sharpe_ratio(r) < 0


class TestSortinoRatio:

    def test_exact_value(self, returns):
        assert abs(sortino_ratio(returns) - 0.6674052533) < 1e-6

    def test_no_downside_returns_zero(self):
        r = pd.Series([0.01, 0.02, 0.03])
        assert sortino_ratio(r) == 0.0


class TestGainToPainRatio:

    def test_exact_value(self, returns):
        assert abs(gain_to_pain_ratio(returns) - 0.0635553078) < 1e-6

    def test_all_positive_returns_inf(self):
        r = pd.Series([0.01, 0.02, 0.03])
        assert gain_to_pain_ratio(r) == float("inf")

    def test_all_negative_returns_negative(self):
        r = pd.Series([-0.01, -0.02, -0.03])
        assert gain_to_pain_ratio(r) < 0


class TestBestWorstPeriods:

    def test_exact_values(self, returns):
        bw = best_worst_periods(returns, window=21)
        assert abs(bw["best"] - 0.2656546398) < 1e-6
        assert abs(bw["worst"] - (-0.1987885737)) < 1e-6
        assert bw["window"] == 21

    def test_best_ge_worst(self, returns):
        bw = best_worst_periods(returns, window=21)
        assert bw["best"] >= bw["worst"]


class TestReturnStatistics:

    def test_exact_values(self, returns):
        stats = return_statistics(returns)
        assert abs(stats["mean_daily"] - 0.0004929112) < 1e-6
        assert abs(stats["std_daily"] - 0.0199831287) < 1e-6
        assert abs(stats["annualized_mean"] - 0.1242136258) < 1e-6
        assert abs(stats["min"] - (-0.0544518551)) < 1e-6
        assert abs(stats["max"] - 0.0549244811) < 1e-6

    def test_has_all_keys(self, returns):
        stats = return_statistics(returns)
        expected_keys = {"mean_daily", "std_daily", "annualized_mean", "annualized_std",
                         "skewness", "kurtosis", "min", "max"}
        assert set(stats.keys()) == expected_keys


class TestEdgeCases:

    def test_empty_returns_sharpe_returns_nan(self):
        result = sharpe_ratio(pd.Series(dtype=float))
        assert np.isnan(result)

    def test_empty_returns_sortino_returns_zero(self):
        # Empty series has no downside returns → 0.0
        assert sortino_ratio(pd.Series(dtype=float)) == 0.0

    def test_single_value_rolling_vol(self):
        r = pd.Series([0.01])
        rv = rolling_volatility(r, window=21)
        assert rv.isna().all()

    def test_all_zero_returns_sortino_zero(self):
        r = pd.Series([0.0] * 100)
        assert sortino_ratio(r) == 0.0
