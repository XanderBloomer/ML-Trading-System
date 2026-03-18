"""
Run the full model training and evaluation pipeline.

Flow:
    1. Load raw OHLCV data for multiple tickers from disk
    2. Engineer features for each ticker independently
    3. Combine into a single pooled DataFrame
    4. Split into train/test by date (no shuffling)
    5. Train Logistic Regression + XGBoost
    6. Evaluate both on out-of-sample test set
    7. Print comparison table + feature importances

WHY POOL MULTIPLE TICKERS:
    A single ticker gives ~170 test samples — too few for reliable
    evaluation. One bad month can swing results by 5%.
    Pooling AAPL + MSFT + SPY gives ~500 test samples, which is
    enough to draw more meaningful conclusions.

    Features are computed per-ticker independently (so AAPL's SMA
    is never contaminated by MSFT's prices), then combined.

Usage:
    poetry run python scripts/run_model_training.py
"""

import pandas as pd

from ml_trading_system.data.data_loader import DataLoader
from ml_trading_system.features.feature_engineer import FeatureEngineer
from ml_trading_system.models.evaluator import ModelEvaluator
from ml_trading_system.models.splitter import TimeSeriesSplitter
from ml_trading_system.models.trainer import ModelTrainer

TICKERS = {
    "AAPL": "aapl_2020-01-01_2024-01-01.csv",
    "MSFT": "msft_2020-01-01_2024-01-01.csv",
    "SPY": "spy_2020-01-01_2024-01-01.csv",
}


def load_and_engineer(
    ticker: str,
    filename: str,
    loader: DataLoader,
    fe: FeatureEngineer,
) -> pd.DataFrame:
    """
    Load raw data for one ticker and compute its features.

    WHY PER-TICKER FEATURE ENGINEERING:
        Features like SMA and RSI are computed from a single price
        series. If we combined raw prices first, AAPL's SMA would
        bleed into MSFT's rows — that's nonsensical.
        We compute features independently, then stack the results.

    Args:
        ticker:   Ticker symbol (used only for logging).
        filename: CSV filename in the data directory.
        loader:   DataLoader instance.
        fe:       FeatureEngineer instance.

    Returns:
        DataFrame of engineered features with DatetimeIndex.
    """
    df_raw = loader.load(filename)
    df_features = fe.compute(df_raw)
    print(f"  {ticker}: {len(df_raw)} raw rows → {len(df_features)} feature rows")
    return df_features


def main() -> None:
    loader = DataLoader()
    fe = FeatureEngineer()

    # --- Step 1: Load and engineer features per ticker ---
    print("Loading and engineering features...")
    frames = []
    for ticker, filename in TICKERS.items():
        df = load_and_engineer(ticker, filename, loader, fe)
        frames.append(df)

    # --- Step 2: Combine into one pooled DataFrame ---
    # pd.concat stacks the DataFrames vertically.
    # The index (dates) will have duplicates — that's fine.
    # We sort by date so the time-based splitter sees rows
    # in chronological order across all tickers.
    df_pooled = pd.concat(frames).sort_index()
    print(f"\nPooled dataset: {len(df_pooled)} total rows")
    print(f"Date range: {df_pooled.index[0].date()} → {df_pooled.index[-1].date()}")
    print(f"Target balance: {df_pooled['target'].mean():.1%} up days\n")

    # --- Step 3: Train/test split ---
    # The splitter splits by date across the whole pooled set.
    # All three tickers' rows from before the split date go into
    # training. All rows after go into test.
    # This correctly simulates training up to a point in time
    # and testing on everything after — regardless of ticker.
    print("Splitting data...")
    splitter = TimeSeriesSplitter(test_size=0.2, gap_days=30)
    split = splitter.split(df_pooled, feature_names=fe.feature_names)
    print(f"  {split.summary()}\n")

    # --- Step 4: Train models ---
    print("Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all(split)
    print("  Logistic Regression ✅")
    print("  XGBoost             ✅\n")

    # --- Step 5: Evaluate ---
    print("Evaluating on out-of-sample test set...")
    evaluator = ModelEvaluator()
    results = []
    for name, model in models.items():
        result = evaluator.evaluate(model, split.X_test, split.y_test)
        results.append(result)
        print(result.summary())

    # --- Step 6: Side-by-side comparison ---
    print("--- Model Comparison ---")
    print(evaluator.compare(results))

    # --- Step 7: Feature importances ---
    print("\n--- Feature Importances ---")
    for name, model in models.items():
        print(f"\n{name}:")
        print(evaluator.feature_importance(model, top_n=13).to_string())


if __name__ == "__main__":
    main()
