"""
Feature engineering for ML trading signals.

All features are computed from raw OHLCV data and are designed
to be strictly point-in-time (no lookahead bias).
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Transforms raw OHLCV price data into ML-ready features.

    Features produced:
        - Log returns (1d, 5d, 10d, 21d)
        - Rolling volatility (5d, 21d)
        - Simple moving averages (10d, 50d)
        - SMA ratio (price / SMA — momentum proxy)
        - RSI (14d)
        - Volume change
        - Target: next-day return direction (1 = up, 0 = down)
    """

    def __init__(self, target_horizon: int = 1) -> None:
        """
        Args:
            target_horizon: Number of days ahead to predict (default: 1).
        """
        self.target_horizon = target_horizon

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from an OHLCV DataFrame.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume].

        Returns:
            DataFrame with features and target column, NaN rows dropped.
        """
        df = df.copy()
        close = df["Close"]
        volume = df["Volume"]

        # Returns
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)
        df["return_21d"] = close.pct_change(21)

        # Log return
        df["log_return_1d"] = np.log(close / close.shift(1))

        # Volatility
        df["volatility_5d"] = df["log_return_1d"].rolling(5).std()
        df["volatility_21d"] = df["log_return_1d"].rolling(21).std()

        # Moving Averages
        df["sma_10"] = close.rolling(10).mean()
        df["sma_50"] = close.rolling(50).mean()

        # Price relative to SMA
        df["price_to_sma10"] = close / df["sma_10"]
        df["price_to_sma50"] = close / df["sma_50"]

        # SMA crossover ratio
        df["sma10_to_sma50"] = df["sma_10"] / df["sma_50"]

        # RSI (14-day)
        df["rsi_14"] = self._compute_rsi(close, period=14)

        # Volume
        df["volume_change_1d"] = volume.pct_change(1)
        df["volume_sma_10"] = volume.rolling(10).mean()
        df["volume_ratio"] = volume / df["volume_sma_10"]

        # Target (must be last — avoids any accidental leakage)
        future_return = close.shift(-self.target_horizon) / close - 1
        df["target"] = (future_return > 0).astype(int)

        # Drop raw price columns (model should only see features)
        feature_cols = [
            "return_1d",
            "return_5d",
            "return_10d",
            "return_21d",
            "log_return_1d",
            "volatility_5d",
            "volatility_21d",
            "price_to_sma10",
            "price_to_sma50",
            "sma10_to_sma50",
            "rsi_14",
            "volume_change_1d",
            "volume_ratio",
            "target",
        ]

        result = df[feature_cols].dropna()
        return result

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, float("inf"))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @property
    def feature_names(self) -> list[str]:
        """Returns the list of feature column names (excludes target)."""
        return [
            "return_1d",
            "return_5d",
            "return_10d",
            "return_21d",
            "log_return_1d",
            "volatility_5d",
            "volatility_21d",
            "price_to_sma10",
            "price_to_sma50",
            "sma10_to_sma50",
            "rsi_14",
            "volume_change_1d",
            "volume_ratio",
        ]
