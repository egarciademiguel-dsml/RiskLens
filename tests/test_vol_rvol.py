import numpy as np
import pandas as pd
import pytest

from src.analytics.vol_rvol import (
    engineer_vol_features,
    build_target,
    fit_rvol,
    predict_current_vol,
)
from src.analytics.monte_carlo import simulate_paths


@pytest.fixture
def market_data():
    """Deterministic close and returns series."""
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 500),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
        name="returns",
    )
    close = 100 * (1 + returns).cumprod()
    close.name = "close"
    return close, returns


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerVolFeatures:

    def test_output_is_dataframe(self, market_data):
        _, returns = market_data
        features = engineer_vol_features(returns)
        assert isinstance(features, pd.DataFrame)

    def test_no_nans(self, market_data):
        _, returns = market_data
        features = engineer_vol_features(returns)
        assert not features.isna().any().any()

    def test_expected_columns(self, market_data):
        _, returns = market_data
        features = engineer_vol_features(returns)
        expected = {"returns", "vol_5d", "vol_10d", "vol_21d", "vol_63d",
                    "mean_ret_5d", "mean_ret_10d", "mean_ret_21d",
                    "skew_21d", "kurtosis_21d", "abs_ret", "sq_ret"}
        assert set(features.columns) == expected

    def test_fewer_rows_than_input(self, market_data):
        _, returns = market_data
        features = engineer_vol_features(returns)
        assert len(features) < len(returns)


# ---------------------------------------------------------------------------
# Build target
# ---------------------------------------------------------------------------

class TestBuildTarget:

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_target_length(self, market_data, horizon):
        _, returns = market_data
        target = build_target(returns, horizon)
        assert len(target) == len(returns)

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_target_has_nans_at_end(self, market_data, horizon):
        _, returns = market_data
        target = build_target(returns, horizon)
        # Last `horizon` values should be NaN due to shift
        assert target.iloc[-1] != target.iloc[-1]  # NaN check

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_target_positive_where_valid(self, market_data, horizon):
        _, returns = market_data
        target = build_target(returns, horizon).dropna()
        assert (target > 0).all()


# ---------------------------------------------------------------------------
# fit_rvol
# ---------------------------------------------------------------------------

class TestFitRvol:

    def test_returns_expected_keys(self, market_data):
        _, returns = market_data
        result = fit_rvol(returns, horizon=21)
        expected = {"model", "scaler", "feature_cols", "horizon", "r2_train", "predicted_vol"}
        assert set(result.keys()) == expected

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_fits_all_horizons(self, market_data, horizon):
        _, returns = market_data
        result = fit_rvol(returns, horizon=horizon)
        assert result["horizon"] == horizon

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_r2_positive(self, market_data, horizon):
        _, returns = market_data
        result = fit_rvol(returns, horizon=horizon)
        assert result["r2_train"] > 0

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_predicted_vol_positive(self, market_data, horizon):
        _, returns = market_data
        result = fit_rvol(returns, horizon=horizon)
        assert result["predicted_vol"] > 0

    def test_feature_cols_non_empty(self, market_data):
        _, returns = market_data
        result = fit_rvol(returns, horizon=21)
        assert len(result["feature_cols"]) > 0


# ---------------------------------------------------------------------------
# predict_current_vol
# ---------------------------------------------------------------------------

class TestPredictCurrentVol:

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_positive(self, market_data, horizon):
        _, returns = market_data
        result = fit_rvol(returns, horizon=horizon)
        vol = predict_current_vol(result, returns)
        assert vol > 0

    def test_returns_float(self, market_data):
        _, returns = market_data
        result = fit_rvol(returns, horizon=21)
        vol = predict_current_vol(result, returns)
        assert isinstance(vol, float)


# ---------------------------------------------------------------------------
# MC integration
# ---------------------------------------------------------------------------

class TestRvolMcIntegration:

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_shape(self, market_data, horizon):
        close, returns = market_data
        rvol_result = fit_rvol(returns, horizon=horizon)
        paths = simulate_paths(
            close, returns, n_days=60, n_simulations=1000,
            seed=42, volatility_model="rvol", rvol_params=rvol_result,
        )
        assert paths.shape == (60, 1000)

    @pytest.mark.parametrize("horizon", [5, 10, 21])
    def test_all_prices_positive(self, market_data, horizon):
        close, returns = market_data
        rvol_result = fit_rvol(returns, horizon=horizon)
        paths = simulate_paths(
            close, returns, n_days=100, n_simulations=5000,
            seed=42, volatility_model="rvol", rvol_params=rvol_result,
        )
        assert (paths > 0).all().all()

    def test_reproducible_with_seed(self, market_data):
        close, returns = market_data
        rvol_result = fit_rvol(returns, horizon=21)
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="rvol", rvol_params=rvol_result)
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="rvol", rvol_params=rvol_result)
        pd.testing.assert_frame_equal(p1, p2)

    def test_rvol_differs_from_constant(self, market_data):
        close, returns = market_data
        rvol_result = fit_rvol(returns, horizon=21)
        p_const = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="constant")
        p_rvol = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                seed=42, volatility_model="rvol", rvol_params=rvol_result)
        assert not p_const.equals(p_rvol)

    def test_rvol_with_t_shocks(self, market_data):
        close, returns = market_data
        rvol_result = fit_rvol(returns, horizon=21)
        paths = simulate_paths(
            close, returns, n_days=60, n_simulations=1000,
            seed=42, distribution="t", df_t=5.0,
            volatility_model="rvol", rvol_params=rvol_result,
        )
        assert paths.shape == (60, 1000)
        assert (paths > 0).all().all()

    def test_auto_fit_when_no_params(self, market_data):
        close, returns = market_data
        paths = simulate_paths(
            close, returns, n_days=30, n_simulations=100,
            seed=42, volatility_model="rvol",
        )
        assert paths.shape == (30, 100)
        assert (paths > 0).all().all()
