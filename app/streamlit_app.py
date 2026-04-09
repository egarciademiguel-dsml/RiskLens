import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data.validate import validate_ticker
from src.data.fetch import fetch_asset_data
from src.data.process import clean_market_data, add_returns
from src.analytics.risk_metrics import (
    annualized_volatility,
    rolling_volatility,
    max_drawdown,
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    gain_to_pain_ratio,
    best_worst_periods,
)
from src.analytics.monte_carlo import (
    simulate_paths,
    compute_var,
    compute_cvar,
    prob_target,
    scenario_buckets,
    simulation_summary,
    fit_t_distribution,
    fit_garch,
)
from src.analytics.backtesting import (
    backtest_var,
    backtest_evt_var,
    backtest_summary,
    constant_fit,
    garch_fit,
    ms_garch_fit,
)

sns.set_theme(style="whitegrid")

# --- Page config ---

st.set_page_config(page_title="RiskLens", layout="wide")
st.title("RiskLens — Investor Risk Assistant")

# --- Sidebar inputs ---

with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Ticker", value="BTC-USD", help="e.g. BTC-USD, AAPL, SPY, GC=F")
    start_date = st.date_input("Start date", value=pd.Timestamp.now() - pd.DateOffset(years=5))
    end_date = st.date_input("End date", value=pd.Timestamp.now())

    st.subheader("Risk Model")
    tier = st.radio(
        "Model Tier",
        options=[
            "Baseline (Constant + Normal)",
            "GARCH + Student-t",
            "MS-GARCH + EVT (Full Model)",
        ],
        index=0,
    )

    with st.expander("What does each tier model?"):
        st.markdown(
            "| Tier | Volatility | Shocks | Captures |\n"
            "|------|-----------|--------|----------|\n"
            "| **Baseline** | Constant | Normal | Symmetric risk, no clustering |\n"
            "| **GARCH+t** | Time-varying | Student-t | Vol clustering + fat tails |\n"
            "| **MS-GARCH+EVT** | Regime-switching GARCH | GPD tails | Regime changes + extreme tails |"
        )

    st.subheader("Simulation")
    _horizon_options = [5, 10, 21, 63, 126, 252, 504, 756, 1260]
    _horizon_labels = ["1w", "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y"]
    _horizon_map = dict(zip(_horizon_labels, _horizon_options))
    horizon_label = st.select_slider("Forecast horizon", options=_horizon_labels, value="1y")
    n_days = _horizon_map[horizon_label]
    n_sims = st.select_slider("Simulations", options=[1000, 5000, 10000, 25000, 50000], value=10000)
    confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95)

    run_analysis = st.button("Analyze", type="primary", use_container_width=True)

# --- Main area ---

if not run_analysis:
    st.info("Configure parameters in the sidebar and click **Analyze** to start.")
    st.stop()

# --- Validation ---

with st.spinner(f"Validating {ticker}..."):
    if not validate_ticker(ticker):
        st.error(f"'{ticker}' is not a valid ticker. Please check and try again.")
        st.stop()

# --- Data fetch & processing ---

with st.spinner(f"Fetching data for {ticker}..."):
    try:
        raw = fetch_asset_data(ticker, start_date=str(start_date), end_date=str(end_date))
    except ValueError as e:
        st.error(str(e))
        st.stop()

df = clean_market_data(raw)
df = add_returns(df)
returns = df["returns"]
close = df["close"]

st.success(f"Loaded {len(df)} trading days for **{ticker}**")

# --- Model fitting & simulation ---

with st.spinner("Running Monte Carlo simulation..."):
    if tier.startswith("Baseline"):
        vol_str = "constant"
        model_kwargs = {}
        model_label = "Baseline (Constant + Normal)"
        garch_info = None
        t_info = None
        ms_info = None
    elif tier.startswith("GARCH"):
        vol_str = "garch"
        garch_info = fit_garch(returns)
        t_info = fit_t_distribution(returns)
        model_kwargs = {"garch_params": garch_info}
        model_label = f"GARCH(1,1) + Student-t (df={t_info['df']:.1f})"
        ms_info = None
    else:  # MS-GARCH + EVT
        from src.analytics.ms_garch import fit_ms_garch
        vol_str = "ms_garch"
        ms_info = fit_ms_garch(returns, n_regimes=2)
        model_kwargs = {"ms_garch_params": ms_info}
        model_label = "MS-GARCH + EVT (2 regimes)"
        garch_info = None
        t_info = None

    paths = simulate_paths(
        close, returns, n_days=n_days, n_simulations=n_sims,
        volatility_model=vol_str,
        **model_kwargs,
    )

    final_prices = paths.iloc[-1]
    initial_price = close.iloc[-1]
    summary = simulation_summary(final_prices, initial_price, confidence=confidence)

# --- Model info boxes ---

if garch_info is not None:
    t_detail = f" | Student-t df = {t_info['df']:.2f} ({t_info['tail_description']})" if t_info else ""
    st.info(
        f"**GARCH(1,1) + Student-t:** "
        f"\u03b1 = {garch_info['alpha']:.4f}, "
        f"\u03b2 = {garch_info['beta']:.4f}, "
        f"persistence = {garch_info['persistence']:.4f} | "
        f"Long-run vol = {garch_info['long_run_vol']:.2%}{t_detail}"
    )

if ms_info is not None:
    regime_names = {0: "Calm", 1: "Crisis"}
    hmm_result = ms_info["hmm_result"]
    current = ms_info["current_regime"]
    parts = []
    for k in range(ms_info["n_regimes"]):
        gp = ms_info["regime_garch"][k]
        gpd = ms_info["regime_gpd"][k]
        mu_k = ms_info["regime_mu"][k]
        marker = " \u2190 current" if k == current else ""
        rname = regime_names.get(k, f"R{k}")
        gpd_str = f"\u03be={gpd['shape']:.3f}" if gpd else "Normal"
        parts.append(
            f"{rname}: \u03c3_lr={gp['long_run_vol']:.2%}, "
            f"persist={gp['persistence']:.3f}, "
            f"tail={gpd_str}{marker}"
        )
    st.info(f"**MS-GARCH + EVT (2 regimes):** " + " | ".join(parts))

col_win, col_lose = st.columns(2)
col_win.metric("Probability of Gain", f"{summary['prob_gain']:.1%}")
col_lose.metric("Probability of Loss", f"{summary['prob_loss']:.1%}")

st.divider()

# --- Key metrics row ---

st.subheader("Key Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ann. Volatility", f"{annualized_volatility(returns):.2%}")
m2.metric("Max Drawdown", f"{max_drawdown(close):.2%}")
m3.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.3f}")
m4.metric("Sortino Ratio", f"{sortino_ratio(returns):.3f}")
m5.metric("Gain-to-Pain", f"{gain_to_pain_ratio(returns):.3f}")

bw = best_worst_periods(returns, window=21)
b1, b2 = st.columns(2)
b1.metric(f"Best {bw['window']}-Day Return", f"{bw['best']:.2%}")
b2.metric(f"Worst {bw['window']}-Day Return", f"{bw['worst']:.2%}")

st.divider()

# --- Scenario distribution ---

st.subheader("Scenario Distribution")

scenarios = scenario_buckets(final_prices, initial_price)
labels = ["Severe Loss\n(<-30%)", "Moderate Loss\n(-30% to -10%)", "Flat\n(-10% to +10%)", "Moderate Gain\n(+10% to +30%)", "Strong Gain\n(>+30%)"]
values = list(scenarios.values())
colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]

fig_sc, ax_sc = plt.subplots(figsize=(10, 3.5))
bars = ax_sc.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, values):
    ax_sc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
               f"{val:.1%}", ha="center", fontweight="bold")
ax_sc.set_ylabel("Probability")
ax_sc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout()
st.pyplot(fig_sc)

st.divider()

# --- Charts in tabs ---

st.subheader("Charts")
tab_price, tab_returns, tab_vol, tab_mc, tab_dist, tab_bt = st.tabs(
    ["Price & Drawdown", "Returns Distribution", "Rolling Volatility", "Monte Carlo Paths", "Final Price Distribution", "VaR Backtest"]
)

with tab_price:
    fig_p, axes_p = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes_p[0].plot(close.index, close.values, color="steelblue", linewidth=0.8)

    # Regime overlay for MS-GARCH
    if ms_info is not None:
        regime_overlay = ms_info["hmm_result"]["regime_labels"]
        n_regimes = ms_info["n_regimes"]
        regime_colors = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
        regime_names_chart = {0: "Calm", 1: "Crisis", 2: "Extreme"}
        dates = close.index[:len(regime_overlay)]
        for r in range(n_regimes):
            mask = regime_overlay == r
            axes_p[0].fill_between(
                dates, close.values[:len(regime_overlay)].min(), close.values[:len(regime_overlay)].max(),
                where=mask, alpha=0.12, color=regime_colors.get(r, "gray"),
                label=regime_names_chart.get(r, f"Regime {r}"))
        axes_p[0].legend(loc="upper left", fontsize=8)

    axes_p[0].set_title(f"{ticker} — Close Price")
    axes_p[0].set_ylabel("Price (USD)")
    dd = drawdown_series(close)
    axes_p[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    axes_p[1].set_title(f"{ticker} — Drawdown (Max: {max_drawdown(close):.2%})")
    axes_p[1].set_ylabel("Drawdown")
    axes_p[1].set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig_p)

with tab_returns:
    fig_r, ax_r = plt.subplots(figsize=(10, 4))
    sns.histplot(returns, bins=100, kde=True, ax=ax_r, color="steelblue")
    ax_r.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
    ax_r.set_title(f"{ticker} — Daily Returns Distribution")
    ax_r.set_xlabel("Daily Return")
    ax_r.legend()
    plt.tight_layout()
    st.pyplot(fig_r)

with tab_vol:
    fig_v, ax_v = plt.subplots(figsize=(12, 4))
    rol_vol = rolling_volatility(returns, window=21)
    ax_v.plot(rol_vol.index, rol_vol.values, color="darkorange", linewidth=0.8)
    ax_v.set_title(f"{ticker} — 21-Day Rolling Annualized Volatility")
    ax_v.set_ylabel("Annualized Volatility")
    ax_v.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig_v)

with tab_mc:
    fig_mc, ax_mc = plt.subplots(figsize=(12, 5))
    sample = paths.iloc[:, :200]
    ax_mc.plot(sample, color="steelblue", alpha=0.03, linewidth=0.5)
    ax_mc.plot(paths.median(axis=1), color="white", linewidth=1.5, label="Median")
    ax_mc.plot(paths.quantile(0.05, axis=1), color="crimson", linewidth=1, linestyle="--", label="5th percentile")
    ax_mc.plot(paths.quantile(0.95, axis=1), color="green", linewidth=1, linestyle="--", label="95th percentile")
    ax_mc.axhline(initial_price, color="orange", linewidth=1, linestyle=":", label=f"Initial: ${initial_price:,.0f}")
    ax_mc.set_title(f"{ticker} — Monte Carlo Paths ({n_sims:,} sims, {n_days} days, {model_label})")
    ax_mc.set_xlabel("Trading Day")
    ax_mc.set_ylabel("Price (USD)")
    ax_mc.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_mc)

with tab_dist:
    var_pct = compute_var(final_prices, initial_price, confidence)
    cvar_pct = compute_cvar(final_prices, initial_price, confidence)
    var_price = initial_price * (1 + var_pct)
    cvar_price = initial_price * (1 + cvar_pct)

    fig_d, ax_d = plt.subplots(figsize=(10, 4))
    sns.histplot(final_prices, bins=100, kde=True, ax=ax_d, color="steelblue", alpha=0.6)
    ax_d.axvline(initial_price, color="orange", linewidth=1.5, linestyle=":", label=f"Initial: ${initial_price:,.0f}")
    ax_d.axvline(var_price, color="crimson", linewidth=1.5, linestyle="--", label=f"VaR {confidence:.0%}: ${var_price:,.0f} ({var_pct:.2%})")
    ax_d.axvline(cvar_price, color="darkred", linewidth=1.5, linestyle="-.", label=f"CVaR {confidence:.0%}: ${cvar_price:,.0f} ({cvar_pct:.2%})")
    ax_d.set_title(f"{ticker} — Final Price Distribution after {n_days} Days")
    ax_d.set_xlabel("Price (USD)")
    ax_d.legend()
    plt.tight_layout()
    st.pyplot(fig_d)

with tab_bt:
    st.markdown("Rolling-window VaR backtest — checks if predicted VaR is consistent with observed losses.")
    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_window = st.slider("Training window (days)", min_value=126, max_value=504, value=252, step=1, key="bt_window")
    with bt_col2:
        bt_step = st.slider("Test every N days", min_value=1, max_value=10, value=5, step=1, key="bt_step")

    if st.button("Run Backtest", key="bt_run"):
        from functools import partial
        if tier.startswith("MS-GARCH"):
            fit_fn = partial(ms_garch_fit, n_regimes=2)
        elif tier.startswith("GARCH"):
            fit_fn = garch_fit
        else:
            fit_fn = constant_fit

        with st.spinner("Running backtest (this may take a minute)..."):
            bt_results = backtest_var(
                close, returns, fit_fn=fit_fn,
                train_window=bt_window, confidence=confidence,
                n_simulations=2000, step=bt_step,
            )

        if len(bt_results) == 0:
            st.warning("Not enough data for backtest with this window size.")
        else:
            bt_sum = backtest_summary(bt_results, confidence=confidence)

            # Metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Observations", bt_sum["n_obs"])
            mc2.metric("Breaches", f"{bt_sum['n_breaches']} ({bt_sum['breach_rate']:.1%})")
            mc3.metric("Kupiec p-value", f"{bt_sum['kupiec']['p_value']:.3f}")
            mc4.metric("Christoffersen p-value", f"{bt_sum['christoffersen']['p_value']:.3f}")

            # Pass/fail
            k_status = "PASS" if bt_sum["kupiec"]["pass"] else "FAIL"
            c_status = "PASS" if bt_sum["christoffersen"]["pass"] else "FAIL"
            expected = bt_sum["expected_rate"]
            st.markdown(
                f"**Expected breach rate:** {expected:.1%} | "
                f"**Observed:** {bt_sum['breach_rate']:.1%} | "
                f"**Kupiec:** {k_status} | **Christoffersen:** {c_status}"
            )

            # Chart: actual returns vs VaR with breach markers
            fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
            ax_bt.plot(bt_results.index, bt_results["actual_return"], color="steelblue", linewidth=0.6, alpha=0.7, label="Actual return")
            ax_bt.plot(bt_results.index, bt_results["predicted_var"], color="crimson", linewidth=1, linestyle="--", label=f"VaR {confidence:.0%}")
            breaches = bt_results[bt_results["breach"]]
            if len(breaches) > 0:
                ax_bt.scatter(breaches.index, breaches["actual_return"], color="red", s=20, zorder=5, label=f"Breaches ({len(breaches)})")
            ax_bt.axhline(0, color="gray", linewidth=0.5)
            ax_bt.set_title(f"{ticker} — VaR Backtest ({model_label}, {confidence:.0%})")
            ax_bt.set_xlabel("Date")
            ax_bt.set_ylabel("Daily Return")
            ax_bt.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_bt)

    # --- EVT backtest ---
    st.markdown("---")
    st.markdown("**EVT (GPD/POT) Backtest** — expanding-window walk-forward validation for Extreme Value Theory VaR.")
    if st.button("Run EVT Backtest", key="evt_bt_run"):
        with st.spinner("Running EVT backtest..."):
            evt_bt = backtest_evt_var(
                returns, confidence=confidence,
                train_window=bt_window, step=bt_step,
            )

        if len(evt_bt) == 0:
            st.warning("Not enough data for EVT backtest with this window size.")
        else:
            evt_sum = backtest_summary(evt_bt, confidence=confidence)

            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Observations", evt_sum["n_obs"])
            ec2.metric("Breaches", f"{evt_sum['n_breaches']} ({evt_sum['breach_rate']:.1%})")
            ec3.metric("Kupiec p-value", f"{evt_sum['kupiec']['p_value']:.3f}")
            ec4.metric("Christoffersen p-value", f"{evt_sum['christoffersen']['p_value']:.3f}")

            ek_status = "PASS" if evt_sum["kupiec"]["pass"] else "FAIL"
            ec_status = "PASS" if evt_sum["christoffersen"]["pass"] else "FAIL"
            st.markdown(
                f"**Expected breach rate:** {evt_sum['expected_rate']:.1%} | "
                f"**Observed:** {evt_sum['breach_rate']:.1%} | "
                f"**Kupiec:** {ek_status} | **Christoffersen:** {ec_status}"
            )

            fig_evt_bt, ax_evt_bt = plt.subplots(figsize=(12, 5))
            ax_evt_bt.plot(evt_bt.index, evt_bt["actual_return"], color="steelblue", linewidth=0.6, alpha=0.7, label="Actual return")
            ax_evt_bt.plot(evt_bt.index, evt_bt["predicted_var"], color="darkgreen", linewidth=1, linestyle="--", label=f"EVT VaR {confidence:.0%}")
            evt_breaches = evt_bt[evt_bt["breach"]]
            if len(evt_breaches) > 0:
                ax_evt_bt.scatter(evt_breaches.index, evt_breaches["actual_return"], color="red", s=20, zorder=5, label=f"Breaches ({len(evt_breaches)})")
            ax_evt_bt.axhline(0, color="gray", linewidth=0.5)
            ax_evt_bt.set_title(f"{ticker} — EVT VaR Backtest ({confidence:.0%})")
            ax_evt_bt.set_xlabel("Date")
            ax_evt_bt.set_ylabel("Daily Return")
            ax_evt_bt.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_evt_bt)

st.divider()

# --- Upside targets ---

st.subheader("Upside Opportunity")
targets = [0.10, 0.25, 0.50, 1.00]
target_data = {f"+{t:.0%}": f"{prob_target(final_prices, initial_price, t):.1%}" for t in targets}
st.dataframe(
    pd.DataFrame(target_data, index=["Probability"]),
    use_container_width=True,
)

# --- Full summary ---

with st.expander("Full Simulation Summary"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Price Outlook**")
        st.write(f"- Initial: ${summary['initial_price']:,.2f}")
        st.write(f"- Mean: ${summary['mean_final_price']:,.2f}")
        st.write(f"- Median: ${summary['median_final_price']:,.2f}")
        st.write(f"- Min: ${summary['min_final_price']:,.2f}")
        st.write(f"- Max: ${summary['max_final_price']:,.2f}")
    with col_b:
        st.write("**Risk**")
        st.write(f"- VaR ({confidence:.0%}): {summary['var']:.2%}")
        st.write(f"- CVaR ({confidence:.0%}): {summary['cvar']:.2%}")
        st.write(f"- Prob of Gain: {summary['prob_gain']:.1%}")
        st.write(f"- Prob of Loss: {summary['prob_loss']:.1%}")
