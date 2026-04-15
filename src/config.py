from pathlib import Path

# Project root: two levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local cache for yfinance fetches (gitignored, local-only — Streamlit Cloud
# fs is ephemeral so caching there is a no-op by design).
DATA_DIR = PROJECT_ROOT / "data"

# Default fetch window
DEFAULT_PERIOD_YEARS = 5

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
