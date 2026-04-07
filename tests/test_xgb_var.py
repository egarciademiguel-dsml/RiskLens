import numpy as np
import pandas as pd
import pytest

from src.analytics.xgb_var import (
    engineer_features,
    fit_quantile_model,
    predict_var,
    backtest_quantile_var,
)


@pytest.fixture
def market_data():
    """Deterministic returns series."""
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 500),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
        name="returns",
    )
    return returns


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerFeatures:

    def test_output_is_dataframe(self, market_data):
        features = engineer_features(market_data)
        assert isinstance(features, pd.DataFrame)

    def test_no_nans(self, market_data):
        features = engineer_features(market_data)
        assert not features.isna().any().any()

    def test_expected_columns(self, market_data):
        features = engineer_features(market_data)
        expected = {"returns", "vol_5d", "vol_10d", "vol_21d", "vol_63d",
                    "mean_ret_5d", "mean_ret_10d", "mean_ret_21d",
                    "skew_21d", "kurtosis_21d", "abs_ret", "sq_ret"}
        assert set(features.columns) == expected

    def test_fewer_rows_than_input(self, market_data):
        features = engineer_features(market_data)
        assert len(features) < len(market_data)


# ---------------------------------------------------------------------------
# fit_quantile_model
# ---------------------------------------------------------------------------

class TestFitQuantileModel:

    def test_returns_expected_keys(self, market_data):
        result = fit_quantile_model(market_data, quantile=0.05)
        expected = {"model", "scaler", "feature_cols", "quantile", "predicted_var"}
        assert set(result.keys()) == expected

    def test_predicted_var_is_float(self, market_data):
        result = fit_quantile_model(market_data, quantile=0.05)
        assert isinstance(result["predicted_var"], float)

    def test_quantile_stored(self, market_data):
        result = fit_quantile_model(market_data, quantile=0.01)
        assert result["quantile"] == 0.01

    def test_5pct_var_is_negative(self, market_data):
        """5th percentile of returns should be a loss (negative)."""
        result = fit_quantile_model(market_data, quantile=0.05)
        assert result["predicted_var"] < 0

    def test_1pct_more_extreme_than_5pct(self, market_data):
        """1% VaR should be a bigger loss than 5% VaR."""
        r01 = fit_quantile_model(market_data, quantile=0.01, seed=42)
        r05 = fit_quantile_model(market_data, quantile=0.05, seed=42)
        assert r01["predicted_var"] < r05["predicted_var"]

    def test_feature_cols_non_empty(self, market_data):
        result = fit_quantile_model(market_data, quantile=0.05)
        assert len(result["feature_cols"]) > 0

    def test_reproducible_with_seed(self, market_data):
        r1 = fit_quantile_model(market_data, quantile=0.05, seed=42)
        r2 = fit_quantile_model(market_data, quantile=0.05, seed=42)
        assert r1["predicted_var"] == r2["predicted_var"]


# ---------------------------------------------------------------------------
# predict_var
# ---------------------------------------------------------------------------

class TestPredictVar:

    def test_returns_float(self, market_data):
        result = fit_quantile_model(market_data, quantile=0.05)
        var = predict_var(result, market_data)
        assert isinstance(var, float)

    def test_matches_fit_prediction(self, market_data):
        """predict_var on same data should match fit's predicted_var."""
        result = fit_quantile_model(market_data, quantile=0.05)
        var = predict_var(result, market_data)
        assert var == pytest.approx(result["predicted_var"], rel=1e-6)


# ---------------------------------------------------------------------------
# backtest_quantile_var
# ---------------------------------------------------------------------------

class TestBacktestQuantileVar:

    def test_returns_dataframe(self, market_data):
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=10)
        assert isinstance(bt, pd.DataFrame)

    def test_expected_columns(self, market_data):
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=10)
        assert set(bt.columns) == {"actual_return", "predicted_var", "breach"}

    def test_breach_is_boolean(self, market_data):
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=10)
        assert bt["breach"].dtype == bool

    def test_var_is_negative(self, market_data):
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=10)
        # Most predicted VaRs should be negative for 5% quantile
        assert (bt["predicted_var"] < 0).mean() > 0.5

    def test_breach_rate_reasonable(self, market_data):
        """Breach rate should be in a reasonable range around the quantile."""
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=5)
        if len(bt) > 0:
            breach_rate = bt["breach"].mean()
            # Allow wide tolerance — small sample, ML model
            assert 0.0 <= breach_rate <= 0.30

    def test_step_reduces_observations(self, market_data):
        bt1 = backtest_quantile_var(market_data, quantile=0.05,
                                    train_window=200, step=1)
        bt5 = backtest_quantile_var(market_data, quantile=0.05,
                                    train_window=200, step=5)
        assert len(bt5) < len(bt1)

    def test_breach_consistency(self, market_data):
        """Breach flag should match actual < predicted."""
        bt = backtest_quantile_var(market_data, quantile=0.05,
                                   train_window=200, step=10)
        expected = bt["actual_return"] < bt["predicted_var"]
        pd.testing.assert_series_equal(bt["breach"], expected, check_names=False)
