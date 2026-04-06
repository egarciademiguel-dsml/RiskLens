import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    st.subheader("Monte Carlo")
    dist_choice = st.radio(
        "Shock Distribution",
        options=["Normal (Gaussian)", "Student-t (fat tails)"],
        index=0,
        help="Student-t captures extreme events better. Degrees of freedom auto-fitted from data.",
    )
    use_t = dist_choice.startswith("Student-t")
    vol_choice = st.radio(
        "Volatility Model",
        options=["Constant", "GARCH(1,1)"],
        index=0,
        help="GARCH captures volatility clustering — high-vol periods follow high-vol periods.",
    )
    use_garch = vol_choice == "GARCH(1,1)"
    n_days = st.slider("Forecast horizon (days)", min_value=30, max_value=504, value=252, step=1)
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

# --- Win / Lose probability ---

with st.spinner("Running Monte Carlo simulation..."):
    dist_str = "t" if use_t else "normal"
    vol_str = "garch" if use_garch else "constant"

    t_info = fit_t_distribution(returns) if use_t else None
    garch_info = fit_garch(returns) if use_garch else None

    paths = simulate_paths(
        close, returns, n_days=n_days, n_simulations=n_sims,
        distribution=dist_str,
        df_t=t_info["df"] if t_info else None,
        volatility_model=vol_str,
        garch_params=garch_info,
    )

    dist_label = f"Student-t (df={t_info['df']:.1f})" if use_t else "Normal"
    vol_label = "GARCH(1,1)" if use_garch else "Constant σ"
    model_label = f"{dist_label}, {vol_label}"

    final_prices = paths.iloc[-1]
    initial_price = close.iloc[-1]
    summary = simulation_summary(final_prices, initial_price, confidence=confidence)

if use_t and t_info is not None:
    st.info(
        f"**Student-t fit:** df = {t_info['df']:.2f} | "
        f"{t_info['tail_description']} | "
        f"Based on {t_info['n_observations']} observations"
    )
if use_garch and garch_info is not None:
    st.info(
        f"**GARCH(1,1) fit:** α = {garch_info['alpha']:.4f}, "
        f"β = {garch_info['beta']:.4f}, "
        f"persistence = {garch_info['persistence']:.4f} | "
        f"Long-run vol = {garch_info['long_run_vol']:.2%}"
    )

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
tab_price, tab_returns, tab_vol, tab_mc, tab_dist = st.tabs(
    ["Price & Drawdown", "Returns Distribution", "Rolling Volatility", "Monte Carlo Paths", "Final Price Distribution"]
)

with tab_price:
    fig_p, axes_p = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes_p[0].plot(close.index, close.values, color="steelblue", linewidth=0.8)
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
