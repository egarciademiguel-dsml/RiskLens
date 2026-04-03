import yfinance as yf


def validate_ticker(ticker: str) -> bool:
    """Check if a ticker is valid by requesting minimal data from yfinance."""
    ticker = ticker.strip().upper()
    if not ticker:
        return False
    data = yf.download(ticker, period="5d", progress=False)
    return len(data) > 0
