"""
Run the full backtest pipeline with confidence threshold comparison.

Flow:
    1. Load raw OHLCV data for each ticker
    2. Engineer features
    3. Split by date (same split as model training)
    4. Train models on training set
    5. For each ticker × model × threshold combination:
       - Generate signals using that threshold
       - Run backtest
       - Record results
    6. Print full comparison table

WHY TEST MULTIPLE THRESHOLDS:
    threshold=0.50 → trade whenever model says "more likely up than down"
    threshold=0.55 → trade only when model is 55%+ confident
    threshold=0.60 → trade only when model is 60%+ confident

    Higher threshold = fewer trades = less commission drag = higher
    quality signals. The cost is fewer trades and more time in cash.
    We run all three to see exactly where the sweet spot is.

Usage:
    poetry run python scripts/run_backtest.py
"""

import pandas as pd

from ml_trading_system.backtesting.backtester import Backtester
from ml_trading_system.data.data_loader import DataLoader
from ml_trading_system.features.feature_engineer import FeatureEngineer
from ml_trading_system.models.splitter import TimeSeriesSplitter
from ml_trading_system.models.trainer import ModelTrainer

TICKERS = {
    "AAPL": "aapl_2020-01-01_2024-01-01.csv",
    "MSFT": "msft_2020-01-01_2024-01-01.csv",
    "SPY": "spy_2020-01-01_2024-01-01.csv",
}

THRESHOLDS = [0.50, 0.55, 0.60]


def main() -> None:
    loader = DataLoader()
    fe = FeatureEngineer()
    backtester = Backtester(commission=0.001)

    # --- Step 1: Load and engineer features per ticker ---
    print("Loading data and engineering features...")
    feature_frames: list[pd.DataFrame] = []
    ticker_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for ticker, filename in TICKERS.items():
        df_raw = loader.load(filename)
        df_features = fe.compute(df_raw)
        prices = df_raw["Close"].reindex(df_features.index)
        ticker_data[ticker] = (df_features, prices)
        feature_frames.append(df_features)
        print(f"  {ticker}: {len(df_features)} feature rows")

    # --- Step 2: Pool and split ---
    df_pooled = pd.concat(feature_frames).sort_index()
    splitter = TimeSeriesSplitter(test_size=0.2, gap_days=30)
    split = splitter.split(df_pooled, feature_names=fe.feature_names)
    print(f"\nSplit: {split.summary()}\n")

    # --- Step 3: Train models ---
    print("Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all(split)
    print("  Logistic Regression ✅")
    print("  XGBoost             ✅\n")

    # --- Step 4: Backtest every ticker × model × threshold ---
    # Collect rows for the summary table
    summary_rows = []

    for ticker in TICKERS:
        df_features, prices = ticker_data[ticker]

        # Filter to test period only
        test_features = df_features[
            (df_features.index >= split.test_start)
            & (df_features.index <= split.test_end)
        ]
        test_prices = prices[test_features.index]
        buy_and_hold = float(test_prices.iloc[-1] / test_prices.iloc[0] - 1)

        print(f"{'='*60}")
        print(f"Ticker: {ticker}  |  Buy & Hold: {buy_and_hold:+.2%}")
        print(f"{'='*60}")

        for model_name, model in models.items():
            print(f"\n  Model: {model_name}")
            print(
                f"  {'Threshold':<12} {'Return':>8} {'vs B&H':>8} "
                f"{'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}"
            )
            print(f"  {'-'*60}")

            for threshold in THRESHOLDS:
                # Generate signals using this threshold
                raw_signals = model.predict_with_threshold(
                    test_features[fe.feature_names],
                    threshold=threshold,
                )
                signals = pd.Series(
                    raw_signals,
                    index=test_features.index,
                    name="signal",
                )

                result = backtester.run(signals, test_prices)
                outperformance = result.total_return - buy_and_hold

                print(
                    f"  {threshold:<12.2f} "
                    f"{result.total_return:>+8.2%} "
                    f"{outperformance:>+8.2%} "
                    f"{result.sharpe_ratio:>8.3f} "
                    f"{result.max_drawdown:>8.2%} "
                    f"{result.n_trades:>7} "
                    f"{result.win_rate:>8.1%}"
                )

                summary_rows.append(
                    {
                        "ticker": ticker,
                        "model": model_name,
                        "threshold": threshold,
                        "return": result.total_return,
                        "buy_and_hold": buy_and_hold,
                        "outperformance": outperformance,
                        "sharpe": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "n_trades": result.n_trades,
                        "win_rate": result.win_rate,
                    }
                )

        print()

    # --- Step 5: Print best result per ticker ---
    print(f"\n{'='*60}")
    print("Best Sharpe per Ticker (across all models + thresholds)")
    print(f"{'='*60}")
    df_summary = pd.DataFrame(summary_rows)
    best = (
        df_summary.sort_values("sharpe", ascending=False)
        .groupby("ticker")
        .first()[["model", "threshold", "return", "sharpe", "max_drawdown", "n_trades"]]
    )
    print(best.to_string())


if __name__ == "__main__":
    main()
