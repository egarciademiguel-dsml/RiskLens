# RiskLens — Investor Risk Assistant

Decision-support tool for financial asset risk assessment. Combines historical risk analytics with Monte Carlo simulation to give investors a clear, quantitative view of downside risk and upside opportunity.

Built with Python and Streamlit. Data sourced from Yahoo Finance via yfinance.

## Features

- **Historical Risk Metrics** — Annualized volatility, max drawdown, Sharpe ratio, Sortino ratio, gain-to-pain ratio, rolling volatility, best/worst periods
- **Monte Carlo Simulation** — GBM-based forward projection with configurable horizon and simulation count
- **Downside Risk** — Value at Risk (VaR) and Conditional VaR (CVaR) at configurable confidence levels
- **Upside Opportunity** — Probability of gain/loss, target return probabilities, five-bucket scenario distribution
- **Interactive Dashboard** — Streamlit app with sidebar configuration, tabbed charts, and full simulation summary
- **75 Automated Tests** — Deterministic fixtures, exact value assertions, edge case coverage

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
│   └── streamlit_app.py        # Interactive dashboard (pure UI, no logic)
├── src/
│   ├── data/                   # Fetch, validate, process, store
│   │   ├── fetch.py            # yfinance data download
│   │   ├── process.py          # OHLCV cleaning, returns computation
│   │   ├── validate.py         # Ticker validation
│   │   └── storage.py          # No-accumulation cache policy
│   ├── analytics/              # Core computation
│   │   ├── risk_metrics.py     # Historical risk functions (Series-in API)
│   │   └── monte_carlo.py      # GBM simulation, VaR, CVaR, scenarios
│   └── config.py               # Project paths and defaults
├── tests/                      # 75 tests across 4 modules
├── notebooks/
│   └── risk_analysis_demo.ipynb  # End-to-end walkthrough
├── docs/                       # Tickets, decisions, workflow
├── data/                       # Raw/processed (gitignored)
├── .github/workflows/          # CI pipeline
├── main.py                     # CLI entry point
├── Makefile                    # Dev commands
├── requirements.txt
└── LICENSE
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Data | pandas, NumPy, yfinance |
| Statistics | SciPy |
| Visualization | matplotlib, seaborn |
| Testing | pytest |
| Quality | Ruff, pre-commit |

## Architecture

All analytics functions follow a **Series-in API**: they take `pd.Series` and return scalars or Series. The Streamlit app is pure integration — it calls analytics functions and renders results, with zero business logic.

Data follows a **no-accumulation policy**: one file per asset maximum, overwrite on refresh, no growing local store.

## Testing

```bash
pytest tests/ -v
```

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_risk_metrics.py` | 20 | Exact values, edge cases, empty series, zero-vol |
| `test_monte_carlo.py` | 18 | VaR/CVaR ordering, seed reproducibility, scenario sums |
| `test_data_pipeline.py` | 15 | Column normalization, NaN handling, returns correctness |
| `test_data_storage.py` | 13 | Overwrite policy, cache operations, cleanup |
| **Total** | **75** | |

All tests are deterministic (seeded fixtures), run offline (mocked yfinance), and use exact value assertions.

## License

MIT
