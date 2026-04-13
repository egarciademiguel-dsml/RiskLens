# RiskLens — Investor Risk Assistant

Decision-support tool for financial asset risk assessment. Combines historical risk analytics, Monte Carlo simulation with multiple volatility models, and statistical backtesting to give investors a clear, quantitative view of downside risk and upside opportunity.

Built with Python and Streamlit. Data sourced from Yahoo Finance via yfinance.

## Features

- **Historical Risk Metrics** — Annualized volatility, max drawdown, Sharpe ratio, Sortino ratio, gain-to-pain ratio, rolling volatility, best/worst periods
- **Monte Carlo Simulation** — 3-tier risk modeling: Baseline (Constant + Normal), GARCH(1,1) + Student-t, MS-GARCH + EVT (Markov-Switching GARCH with per-regime EVT tails)
- **Model-Owned Innovations** — Each volatility model generates its own shock distribution internally (Normal, Student-t, or EVT/GPD)
- **Extreme Value Theory** — GPD/POT tail risk estimation, EVT VaR and CVaR for theoretically justified tail quantiles
- **ML VaR (XGBoost)** — Conditional quantile regression for nonparametric VaR — audits parametric models without distributional assumptions
- **Downside Risk** — Value at Risk (VaR) and Conditional VaR (CVaR) at configurable confidence levels
- **Upside Opportunity** — Probability of gain/loss, target return probabilities, five-bucket scenario distribution
- **VaR Backtesting** — Rolling-window validation with Kupiec unconditional coverage and Christoffersen independence tests
- **Model Comparison** — Programmatic dual ranking: best calibrated vs most conservative model
- **Interactive Dashboard** — Streamlit app with sidebar configuration, tabbed charts, backtesting, and full simulation summary
- **271 Automated Tests** — Deterministic fixtures, exact value assertions, edge case coverage

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd RiskLens
python -m venv .venv
source .venv/Scripts/activate    # Windows
# source .venv/bin/activate      # Mac/Linux
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# CLI data ingestion
python main.py --ticker BTC-USD
```

Or use the Makefile:

```bash
make setup && make install
make app       # Launch Streamlit
make test      # Run test suite
```

## Project Structure

```
RiskLens/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (pure UI, no logic)
├── src/
│   ├── data/                     # Fetch, validate, process, store
│   │   ├── fetch.py              # yfinance data download
│   │   ├── process.py            # OHLCV cleaning, returns computation
│   │   ├── validate.py           # Ticker validation
│   │   └── storage.py            # No-accumulation cache policy
│   ├── analytics/                # Core computation
│   │   ├── risk_metrics.py       # Historical risk functions (Series-in API)
│   │   ├── monte_carlo.py        # Simulation dispatcher, VaR, CVaR, scenarios
│   │   ├── backtesting.py        # Rolling VaR backtest, Kupiec, Christoffersen, model comparison
│   │   ├── vol_constant.py       # Constant volatility (GBM baseline)
│   │   ├── vol_garch.py          # GARCH(1,1) time-varying volatility
│   │   ├── regime_hmm.py         # HMM regime detection (hmmlearn)
│   │   ├── regime_gmm.py         # GMM+RF regime classification (sklearn)
│   │   ├── ms_garch.py           # Markov-Switching GARCH + per-regime EVT tails
│   │   ├── evt.py                # Extreme Value Theory — GPD/POT tail risk
│   │   └── xgb_var.py            # XGBoost conditional quantile regression (nonparametric VaR)
│   └── config.py                 # Project paths, defaults, constants
├── tests/                        # 271 tests across 9 modules
├── notebooks/
│   ├── risk_analysis_walkthrough.ipynb   # Guided tour, 11 sections (includes EVT + XGB quantile)
│   ├── model_comparison.ipynb            # Controlled comparison: 3 tiers x 3 assets, tail vs vol isolation, seed robustness, MS-GARCH internals
│   ├── validation_backtesting.ipynb      # EVT + XGBoost non-MC audit; rolling-window backtest of the 3 tiers
│   ├── model_diagnosis.ipynb             # Why MS-GARCH fails/passes; failure-mode taxonomy; overall conclusions
│   └── horizon_crossover.ipynb           # Horizon at which each tier's ranking flips — drives app recommendation
├── docs/                         # Tickets, decisions, workflow, conclusions, assumptions
├── data/                         # Raw/processed (gitignored)
├── .github/workflows/            # CI pipeline
├── main.py                       # CLI entry point
├── Makefile                      # Dev commands
├── requirements.txt
└── LICENSE
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Data | pandas, NumPy, yfinance |
| Statistics | SciPy (EVT/GPD, distributions), arch (GARCH) |
| ML | scikit-learn, XGBoost, hmmlearn |
| Visualization | matplotlib, seaborn |
| Testing | pytest |
| Quality | Ruff, pre-commit |

## Architecture

All analytics functions follow a **Series-in API**: they take `pd.Series` and return scalars or Series. The Streamlit app is pure integration — it calls analytics functions and renders results, with zero business logic.

Volatility models follow a **dispatcher pattern**: `simulate_paths()` delegates to model-specific `generate_log_returns()` functions via a registry. Each model owns its own module and its own innovation distribution. Adding a new model = new module + one registry line. The app exposes 3 tiers of increasing complexity: Baseline (Constant + Normal), GARCH + Student-t, and MS-GARCH + EVT (Markov-Switching GARCH with per-regime GARCH parameters and per-regime GPD tail distributions).

Two non-MC risk modules sit alongside the simulation pipeline: **EVT** (GPD/POT for theoretically justified tail risk) and **XGBoost conditional quantile regression** (nonparametric VaR that audits parametric models). ML validates — it doesn't replace.

Data follows a **no-accumulation policy**: one file per asset maximum, overwrite on refresh, no growing local store.

## Findings & Assumptions

- **[`docs/conclusions.md`](docs/conclusions.md)** — full prose synthesis of every finding across the deep-dive notebooks: horizon dependence, 3-tier head-to-head, seed robustness, EVT/XGBoost audit, backtest results, MS-GARCH diagnosis, and the failure-mode taxonomy ("each tier fails differently").
- **[`docs/assumptions.md`](docs/assumptions.md)** — consolidated list of modeling assumptions behind every result: data (log-returns, fit window, survivorship), GARCH (stationarity, symmetry, Student-t df as stability indicator), MS-GARCH unified spec (shared persistence, per-regime ω, per-regime GPD), EVT (threshold, iid exceedances), Monte Carlo (seed sensitivity, model-owned innovations), backtesting (Kupiec/Christoffersen), HMM (2 regimes, Gaussian emissions).

## Testing

```bash
pytest tests/ -v
```

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_monte_carlo.py` | 46 | VaR/CVaR ordering, seed reproducibility, cross-model properties |
| `test_regime_hmm.py` | 34 | Fit, predict, transition matrices, regime sorting, MC integration |
| `test_regime_gmm.py` | 33 | GMM+RF fit, regime prediction, MC integration |
| `test_backtesting.py` | 34 | Kupiec, Christoffersen, backtest engine, model comparison, MS-GARCH |
| `test_risk_metrics.py` | 25 | Exact values, edge cases, empty series, zero-vol |
| `test_ms_garch.py` | 23 | Fit, EVT shocks, fallbacks, integration, backtest compatibility |
| `test_evt.py` | 22 | GPD fit, EVT VaR/CVaR, Normal baseline, edge cases |
| `test_xgb_var.py` | 27 | Features, quantile model, prediction, walk-forward backtest |
| `test_data_pipeline.py` | 15 | Column normalization, NaN handling, returns correctness |
| `test_data_storage.py` | 12 | Overwrite policy, cache operations, cleanup |
| **Total** | **271** | |

All tests are deterministic (seeded fixtures), run offline (mocked yfinance), and use exact value assertions.

## License

MIT
