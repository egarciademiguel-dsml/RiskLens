from pathlib import Path

# Project root: two levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default fetch window
DEFAULT_PERIOD_YEARS = 5

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
