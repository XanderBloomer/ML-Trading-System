"""
Tests for the FeatureEngineer class.
"""

import numpy as np
import pandas as pd

from ml_trading_system.features.feature_engineer import FeatureEngineer


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    index = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_compute_returns_expected_columns():
    fe = FeatureEngineer()
    df = fe.compute(_make_ohlcv())
    for col in fe.feature_names:
        assert col in df.columns, f"Missing feature: {col}"
    assert "target" in df.columns


def test_no_nans_after_compute():
    fe = FeatureEngineer()
    df = fe.compute(_make_ohlcv(200))
    assert not df.isnull().any().any(), "Features contain NaN values"


def test_target_is_binary():
    fe = FeatureEngineer()
    df = fe.compute(_make_ohlcv(200))
    assert set(df["target"].unique()).issubset({0, 1})


def test_row_count_reduced_by_warmup():
    """Longest lookback is 50 (sma_50) + 1 target shift = at least 51 rows removed."""
    n = 200
    fe = FeatureEngineer()
    df = fe.compute(_make_ohlcv(n))
    assert len(df) < n


def test_feature_names_match_columns():
    fe = FeatureEngineer()
    df = fe.compute(_make_ohlcv(200))
    for name in fe.feature_names:
        assert name in df.columns
