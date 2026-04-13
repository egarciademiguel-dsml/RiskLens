import argparse

from src.data.validate import validate_ticker
from src.data.fetch import fetch_asset_data
from src.data.process import clean_market_data, add_returns
from src.data.storage import save_asset_data, has_cached_data, cleanup_all_cached_data


def run(ticker: str):
    print(f"Validating ticker: {ticker}")
    if not validate_ticker(ticker):
        print(f"ERROR: '{ticker}' is not a valid ticker.")
        return

    if has_cached_data(ticker):
        print(f"Existing data for {ticker} will be overwritten (no-accumulation policy).")

    print(f"Fetching data for {ticker}...")
    raw = fetch_asset_data(ticker)

    print("Processing...")
    df = clean_market_data(raw)
    df = add_returns(df)

    out_path = save_asset_data(df, ticker)
    print(f"Saved to {out_path}")

    print("\nPreview:")
    print(df.head(10).to_string())


def purge():
    count = cleanup_all_cached_data()
    print(f"Removed {count} cached file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiskLens — fetch asset data")
    parser.add_argument("--ticker", type=str, help="Asset ticker (e.g. BTC-USD)")
    parser.add_argument("--purge", action="store_true", help="Remove all cached data files")
    args = parser.parse_args()

    if args.purge:
        purge()
    elif args.ticker:
        run(args.ticker)
    else:
        parser.print_help()
