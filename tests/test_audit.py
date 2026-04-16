"""Tests for model audit module."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.audit import (
    detect_disagreements,
    regime_conditional_breach_rates,
    comparative_var_table,
)


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=100, freq="D")


@pytest.fixture
def parametric_var(dates):
    np.random.seed(0)
    return pd.Series(-0.02 + np.random.normal(0, 0.003, 100), index=dates, name="parametric")


@pytest.fixture
def xgb_var(dates):
    np.random.seed(1)
    return pd.Series(-0.025 + np.random.normal(0, 0.005, 100), index=dates, name="xgb")


class TestDetectDisagreements:

    def test_returns_dataframe(self, parametric_var, xgb_var):
        result = detect_disagreements(parametric_var, xgb_var)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, parametric_var, xgb_var):
        result = detect_disagreements(parametric_var, xgb_var)
        expected = {"parametric_var", "xgb_var", "abs_diff", "rel_diff", "xgb_deeper"}
        assert set(result.columns) == expected

    def test_only_exceeding_threshold(self, parametric_var, xgb_var):
        result = detect_disagreements(parametric_var, xgb_var, threshold_pct=0.10)
        assert (result["rel_diff"] > 0.10).all()

    def test_higher_threshold_fewer_rows(self, parametric_var, xgb_var):
        r10 = detect_disagreements(parametric_var, xgb_var, threshold_pct=0.10)
        r50 = detect_disagreements(parametric_var, xgb_var, threshold_pct=0.50)
        assert len(r50) <= len(r10)

    def test_xgb_deeper_flag(self, parametric_var, xgb_var):
        result = detect_disagreements(parametric_var, xgb_var, threshold_pct=0.0)
        deeper = result[result["xgb_deeper"]]
        assert (deeper["xgb_var"] < deeper["parametric_var"]).all()


class TestRegimeConditionalBreachRates:

    def test_returns_dataframe(self, dates):
        bt = pd.DataFrame({"breach": [True, False] * 50}, index=dates)
        regimes = pd.Series([0] * 60 + [1] * 40, index=dates)
        result = regime_conditional_breach_rates(bt, regimes)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, dates):
        bt = pd.DataFrame({"breach": [True, False] * 50}, index=dates)
        regimes = pd.Series([0] * 60 + [1] * 40, index=dates)
        result = regime_conditional_breach_rates(bt, regimes)
        assert set(result.columns) == {"n_obs", "n_breaches", "breach_rate"}

    def test_total_matches(self, dates):
        bt = pd.DataFrame({"breach": [True, False] * 50}, index=dates)
        regimes = pd.Series([0] * 60 + [1] * 40, index=dates)
        result = regime_conditional_breach_rates(bt, regimes)
        assert result["n_obs"].sum() == 100

    def test_breach_rate_range(self, dates):
        bt = pd.DataFrame({"breach": [True, False] * 50}, index=dates)
        regimes = pd.Series([0] * 60 + [1] * 40, index=dates)
        result = regime_conditional_breach_rates(bt, regimes)
        assert (result["breach_rate"] >= 0).all()
        assert (result["breach_rate"] <= 1).all()


class TestComparativeVarTable:

    def test_returns_dataframe(self, dates):
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        var_a = pd.Series(-0.02 * np.ones(100), index=dates)
        result = comparative_var_table(returns, {"ModelA": var_a})
        assert isinstance(result, pd.DataFrame)

    def test_has_breach_columns(self, dates):
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        var_a = pd.Series(-0.02 * np.ones(100), index=dates)
        result = comparative_var_table(returns, {"ModelA": var_a})
        assert "breach_modela" in result.columns
        assert "var_modela" in result.columns

    def test_multiple_models(self, dates):
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        var_a = pd.Series(-0.02 * np.ones(100), index=dates)
        var_b = pd.Series(-0.03 * np.ones(100), index=dates)
        result = comparative_var_table(returns, {"ModelA": var_a, "ModelB": var_b})
        assert "var_modela" in result.columns
        assert "var_modelb" in result.columns
