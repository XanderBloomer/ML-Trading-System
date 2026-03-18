"""
Run the data ingestion pipeline.
Usage: poetry run python scripts/run_data_pipeline.py
"""

from ml_trading_system.data.data_loader import DataLoader


def main() -> None:
    loader = DataLoader()

    tickers = ["AAPL", "MSFT", "SPY"]
    start = "2020-01-01"
    end = "2024-01-01"

    for ticker in tickers:
        loader.fetch_and_save(ticker, start=start, end=end)

    print("\nData pipeline complete ✅")


if __name__ == "__main__":
    main()
