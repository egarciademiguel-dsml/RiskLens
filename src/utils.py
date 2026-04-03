from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def ensure_dirs():
    """Create data directories if they don't exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
