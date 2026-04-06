"""Tests for VaR backtesting module."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.backtesting import (
    kupiec_test,
    christoffersen_test,
    backtest_var,
    backtest_summary,
    constant_fit,
    garch_fit,
)


# ---------------------------------------------------------------------------
# Kupiec test
# ---------------------------------------------------------------------------

class TestKupiec:
    def test_perfect_calibration(self):
        """Breach rate exactly matches expected → high p-value, pass."""
        result = kupiec_test(n_obs=1000, n_breaches=50, confidence=0.95)
        assert result["expected_rate"] == pytest.approx(0.05)
        assert result["observed_rate"] == 0.05
        assert result["pass"] == True
        assert result["p_value"] > 0.90

    def test_too_many_breaches(self):
        """Way too many breaches → reject."""
        result = kupiec_test(n_obs=1000, n_breaches=150, confidence=0.95)
        assert result["observed_rate"] == 0.15
        assert result["pass"] == False
        assert result["p_value"] < 0.05

    def test_too_few_breaches(self):
        """Almost no breaches → overly conservative, reject."""
        result = kupiec_test(n_obs=1000, n_breaches=5, confidence=0.95)
        assert result["observed_rate"] == 0.005
        assert result["pass"] == False

    def test_zero_breaches(self):
        """No breaches at all → should not crash."""
        result = kupiec_test(n_obs=500, n_breaches=0, confidence=0.95)
        assert result["observed_rate"] == 0.0
        assert result["pass"] == False
        assert np.isfinite(result["lr_statistic"])

    def test_all_breaches(self):
        """All observations are breaches → should not crash."""
        result = kupiec_test(n_obs=100, n_breaches=100, confidence=0.95)
        assert result["observed_rate"] == 1.0
        assert result["pass"] == False

    def test_99_confidence(self):
        """99% VaR → expected rate 1%."""
        result = kupiec_test(n_obs=1000, n_breaches=10, confidence=0.99)
        assert result["expected_rate"] == pytest.approx(0.01)
        assert result["pass"] == True

    def test_return_keys(self):
        result = kupiec_test(100, 5, 0.95)
        assert set(result.keys()) == {"expected_rate", "observed_rate", "lr_statistic", "p_value", "pass"}


# ---------------------------------------------------------------------------
# Christoffersen test
# ---------------------------------------------------------------------------

class TestChristoffersen:
    def test_independent_breaches(self):
        """Randomly scattered breaches → pass."""
        rng = np.random.RandomState(42)
        breaches = (rng.rand(500) < 0.05).astype(int)
        result = christoffersen_test(breaches)
        assert result["pass"] == True

    def test_clustered_breaches(self):
        """All breaches in one block → fail independence."""
        breaches = np.zeros(500, dtype=int)
        breaches[100:130] = 1  # 30 consecutive breaches
        result = christoffersen_test(breaches)
        assert result["pass"] == False

    def test_no_breaches(self):
        """No breaches → trivially independent."""
        breaches = np.zeros(200, dtype=int)
        result = christoffersen_test(breaches)
        assert result["pass"] == True

    def test_all_breaches(self):
        """All breaches → trivially independent (no transitions from 0)."""
        breaches = np.ones(100, dtype=int)
        result = christoffersen_test(breaches)
        assert result["pass"] == True

    def test_single_observation(self):
        """Edge case: 1 observation."""
        result = christoffersen_test(np.array([1]))
        assert result["pass"] == True

    def test_return_keys(self):
        result = christoffersen_test(np.array([0, 1, 0, 1]))
        assert set(result.keys()) == {"lr_statistic", "p_value", "pass"}

    def test_alternating_pattern(self):
        """Perfectly alternating 0-1 → should detect dependence."""
        breaches = np.tile([0, 1], 100)
        result = christoffersen_test(breaches)
        # 50% breach rate with perfect alternation → strong dependence
        assert result["pass"] == False


# ---------------------------------------------------------------------------
# Backtest on synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_data(n=600, seed=42):
    """Create synthetic close/returns for backtest tests."""
    rng = np.random.RandomState(seed)
    returns = pd.Series(rng.normal(0.0005, 0.02, n), name="returns")
    prices = 100 * np.exp(np.cumsum(returns.values))
    close = pd.Series(prices, name="close")
    # Add date index
    dates = pd.bdate_range("2020-01-01", periods=n)
    close.index = dates
    returns.index = dates
    return close, returns


class TestBacktestVar:
    def test_returns_dataframe(self):
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        assert isinstance(results, pd.DataFrame)
        assert "actual_return" in results.columns
        assert "predicted_var" in results.columns
        assert "breach" in results.columns

    def test_breach_is_boolean(self):
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        assert results["breach"].dtype == bool

    def test_var_is_negative(self):
        """At 95% confidence, VaR should be negative (a loss threshold)."""
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        assert (results["predicted_var"] < 0).all()

    def test_step_reduces_observations(self):
        close, returns = _make_synthetic_data()
        r1 = backtest_var(close, returns, train_window=252, step=1, n_simulations=500)
        r5 = backtest_var(close, returns, train_window=252, step=5, n_simulations=500)
        assert len(r5) < len(r1)

    def test_garch_fit_runs(self):
        """GARCH fit function works in backtest."""
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, fit_fn=garch_fit, train_window=252, step=20, n_simulations=500)
        assert len(results) > 0

    def test_breach_consistency(self):
        """Breach flag matches actual < var comparison."""
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        expected_breach = results["actual_return"] < results["predicted_var"]
        pd.testing.assert_series_equal(results["breach"], expected_breach, check_names=False)


# ---------------------------------------------------------------------------
# Backtest summary
# ---------------------------------------------------------------------------

class TestBacktestSummary:
    def test_summary_keys(self):
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        summary = backtest_summary(results, confidence=0.95)
        assert "n_obs" in summary
        assert "breach_rate" in summary
        assert "kupiec" in summary
        assert "christoffersen" in summary

    def test_breach_count_matches(self):
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        summary = backtest_summary(results)
        assert summary["n_breaches"] == results["breach"].sum()
        assert summary["n_obs"] == len(results)

    def test_expected_rate(self):
        close, returns = _make_synthetic_data()
        results = backtest_var(close, returns, train_window=252, step=10, n_simulations=500)
        summary = backtest_summary(results, confidence=0.99)
        assert summary["expected_rate"] == pytest.approx(0.01)
