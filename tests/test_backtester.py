"""
Tests for the Backtester class.
"""

import numpy as np
import pandas as pd
import pytest

from ml_trading_system.backtesting.backtester import Backtester


def _make_prices(n: int = 100, start: str = "2023-01-01") -> pd.Series:
    """Generate a synthetic price series."""
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.cumprod(1 + returns)
    index = pd.date_range(start, periods=n, freq="B")
    return pd.Series(prices, index=index, name="Close")


def _make_signals(prices: pd.Series, always_on: bool = True) -> pd.Series:
    """Generate a constant signal (always in or always out)."""
    value = 1 if always_on else 0
    return pd.Series(value, index=prices.index, name="signal")


def test_always_in_matches_buy_and_hold():
    """With signal=1 every day and no commission, strategy ≈ buy-and-hold."""
    backtester = Backtester(commission=0.0)
    prices = _make_prices()
    signals = _make_signals(prices, always_on=True)
    result = backtester.run(signals, prices)

    # Total return should be very close to buy-and-hold
    # (small diff due to shift(1) dropping the first day)
    assert abs(result.total_return - result.benchmark_return) < 0.01


def test_always_out_returns_near_zero():
    """With signal=0 every day, strategy should return ~0 (just costs)."""
    backtester = Backtester(commission=0.001)
    prices = _make_prices()
    signals = _make_signals(prices, always_on=False)
    result = backtester.run(signals, prices)

    # No positions held = no market exposure = near-zero return
    assert abs(result.total_return) < 0.01


def test_commission_reduces_return():
    """Strategy with commission should return less than without."""
    prices = _make_prices()
    # Alternating signal creates maximum number of trades = maximum cost
    signals = pd.Series(
        [1, 0] * (len(prices) // 2),
        index=prices.index[: len(prices) // 2 * 2],
    )
    result_no_cost = Backtester(commission=0.0).run(signals, prices)
    result_with_cost = Backtester(commission=0.001).run(signals, prices)

    assert result_with_cost.total_return < result_no_cost.total_return


def test_equity_curve_starts_at_one():
    """Equity curve first value should be close to 1.0.

    WHY < 0.01 tolerance (not 1e-6):
        On day 1, a position change from NaN → 1 triggers commission.
        This means the first equity value is slightly below 1.0 by
        exactly the commission amount. The backtester is correct —
        the test tolerance reflects this real behaviour.
    """
    backtester = Backtester()
    prices = _make_prices()
    signals = _make_signals(prices)
    result = backtester.run(signals, prices)

    assert abs(result.equity_curve.iloc[0] - 1.0) < 0.01


def test_max_drawdown_is_negative():
    """Max drawdown should always be <= 0."""
    backtester = Backtester()
    prices = _make_prices()
    signals = _make_signals(prices)
    result = backtester.run(signals, prices)

    assert result.max_drawdown <= 0.0


def test_win_rate_between_zero_and_one():
    """Win rate must be a valid fraction."""
    backtester = Backtester()
    prices = _make_prices()
    signals = pd.Series(
        np.random.randint(0, 2, len(prices)),
        index=prices.index,
    )
    result = backtester.run(signals, prices)

    assert 0.0 <= result.win_rate <= 1.0


def test_mismatched_dates_raises():
    """Signals and prices with no overlapping dates should raise."""
    backtester = Backtester()
    prices = _make_prices(start="2023-01-01")
    signals = _make_signals(_make_prices(start="2024-06-01"))

    with pytest.raises(ValueError, match="No overlapping dates"):
        backtester.run(signals, prices)


def test_negative_commission_raises():
    """Negative commission is not valid."""
    with pytest.raises(ValueError):
        Backtester(commission=-0.001)
