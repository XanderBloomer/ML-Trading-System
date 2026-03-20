"""
Backtesting engine for evaluating trading signals.

WHY BACKTESTING MATTERS MORE THAN MODEL ACCURACY:
    A model with 56% directional accuracy sounds weak.
    But if those correct predictions are on bigger moves,
    and wrong predictions are on smaller moves, the strategy
    can be very profitable. Accuracy doesn't capture this.

    The backtester simulates actual trading and measures
    what actually matters — risk-adjusted returns.

ASSUMPTIONS (deliberately simple for v1):
    - Signal = 1 → fully invested (100% of capital)
    - Signal = 0 → hold cash (0% invested)
    - Trades execute at next day's open (not same-day close)
    - Transaction cost applied on every position change
    - No leverage, no short selling
    - Single asset at a time

    These assumptions will be relaxed in later phases
    (portfolio construction, position sizing, shorting).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Holds the full output of a single backtest run.

    Attributes:
        equity_curve:      Daily portfolio value (starting from 1.0).
        strategy_returns:  Daily returns of the strategy.
        benchmark_returns: Daily returns of buy-and-hold.
        trades:            DataFrame recording every trade entry/exit.
        total_return:      Total strategy return over the period.
        benchmark_return:  Total buy-and-hold return over the same period.
        sharpe_ratio:      Annualised Sharpe ratio of the strategy.
        max_drawdown:      Worst peak-to-trough loss (negative number).
        win_rate:          Fraction of trades that were profitable.
        n_trades:          Total number of trades executed.
        avg_holding_days:  Average number of days per trade.
    """

    equity_curve: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    trades: pd.DataFrame
    total_return: float
    benchmark_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_holding_days: float

    def summary(self) -> str:
        """Print a clean performance summary."""
        outperformance = self.total_return - self.benchmark_return
        idx = self.equity_curve.index
        start = str(idx[0])[:10]
        end = str(idx[-1])[:10]
        return (
            f"\n{'='*50}\n"
            f"Backtest Results\n"
            f"{'='*50}\n"
            f"Period         : {start} → {end}\n"
            f"\n--- Returns ---\n"
            f"Strategy       : {self.total_return:+.2%}\n"
            f"Buy & Hold     : {self.benchmark_return:+.2%}\n"
            f"Outperformance : {outperformance:+.2%}\n"
            f"\n--- Risk ---\n"
            f"Sharpe Ratio   : {self.sharpe_ratio:.3f}  "
            f"(>1.0 good, >2.0 excellent)\n"
            f"Max Drawdown   : {self.max_drawdown:.2%}\n"
            f"\n--- Trading ---\n"
            f"Total Trades   : {self.n_trades}\n"
            f"Win Rate       : {self.win_rate:.1%}\n"
            f"Avg Hold (days): {self.avg_holding_days:.1f}\n"
            f"{'='*50}\n"
        )


class Backtester:
    """
    Simulates a long-only strategy driven by model signals.

    Args:
        commission: Round-trip transaction cost as a fraction.
                    Default 0.001 = 0.1% per trade (one way).
                    Applied when entering AND exiting a position.

    WHY 0.1% COMMISSION:
        This is realistic for retail brokers (Interactive Brokers,
        Alpaca, etc.). It's conservative enough that any strategy
        surviving this cost has real edge. Ignoring costs is the
        most common backtesting mistake.

    Usage:
        backtester = Backtester(commission=0.001)
        result = backtester.run(signals, prices)
        print(result.summary())
    """

    def __init__(self, commission: float = 0.001) -> None:
        if commission < 0:
            raise ValueError(f"Commission must be >= 0, got {commission}")
        self.commission = commission

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
    ) -> BacktestResult:
        """
        Run a backtest given daily signals and prices.

        Args:
            signals: pd.Series with DatetimeIndex.
                     Values: 1 = hold long position, 0 = hold cash.
                     Index must align with prices index.
            prices:  pd.Series of daily closing prices with DatetimeIndex.
                     Must cover at least the full signal period.

        Returns:
            BacktestResult with equity curve, metrics, and trade log.

        WHY signals AND prices are separate inputs:
            The model produces signals from features. The features
            are computed from prices. But the backtester should not
            know about features — it only cares about:
            "given this signal, what return did I get?"
            Keeping them separate enforces clean separation of concerns.
        """
        # --- Align signals and prices ---
        # Both series must cover the same dates.
        # We inner-join to handle any date mismatches cleanly.
        df = (
            pd.DataFrame(
                {
                    "signal": signals,
                    "price": prices,
                }
            )
            .dropna()
            .sort_index()
        )

        if len(df) == 0:
            raise ValueError("No overlapping dates between signals and prices.")

        # --- Compute daily returns ---
        # Return on day t = (price[t] / price[t-1]) - 1
        # shift(1) = yesterday's price → no lookahead ✅
        df["daily_return"] = df["price"].pct_change()

        # --- Apply signals ---
        # Strategy return on day t = signal[t-1] * daily_return[t]
        #
        # WHY signal[t-1] (shift(1)):
        #   The signal is generated at end of day t-1 using data
        #   available at t-1. The trade executes at open of day t.
        #   We capture the return from open to close of day t.
        #   Using signal[t] directly would be lookahead bias —
        #   we'd be acting on information we don't yet have.
        df["position"] = df["signal"].shift(1)

        # --- Apply transaction costs ---
        # A trade occurs when the position changes (0→1 or 1→0).
        # Each change incurs commission on both sides of the trade.
        #
        # WHY abs(diff()) > 0:
        #   diff() gives the change in position each day.
        #   abs() handles both entries (0→1 = +1) and exits (1→0 = -1).
        #   commission is subtracted from that day's return.
        df["trade_occurred"] = df["position"].diff().abs().fillna(0)
        df["cost"] = df["trade_occurred"] * self.commission

        # Strategy return after costs
        df["strategy_return"] = df["position"] * df["daily_return"] - df["cost"]

        df = df.dropna()

        # --- Build equity curve ---
        # Equity curve starts at 1.0 and compounds daily.
        # cumprod() of (1 + daily_return) gives the growth factor.
        equity_curve = (1 + df["strategy_return"]).cumprod()
        benchmark_curve = (1 + df["daily_return"]).cumprod()

        # --- Compute metrics ---
        total_return = float(equity_curve.iloc[-1] - 1)
        benchmark_return = float(benchmark_curve.iloc[-1] - 1)
        sharpe = self._sharpe_ratio(df["strategy_return"])
        max_dd = self._max_drawdown(equity_curve)
        trades = self._build_trade_log(df)
        win_rate = self._win_rate(trades)
        avg_hold = float(trades["holding_days"].mean()) if len(trades) > 0 else 0.0

        return BacktestResult(
            equity_curve=equity_curve,
            strategy_returns=df["strategy_return"],
            benchmark_returns=df["daily_return"],
            trades=trades,
            total_return=total_return,
            benchmark_return=benchmark_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=len(trades),
            avg_holding_days=avg_hold,
        )

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Compute the annualised Sharpe ratio.

        WHY 252:
            There are ~252 trading days per year for US equities.
            We annualise by multiplying daily Sharpe by sqrt(252).
            This is the industry standard.

        WHY we don't subtract a risk-free rate:
            At this stage we use excess return = 0 as the risk-free
            rate. In a later phase we'll subtract the actual T-bill
            rate. The difference is small but matters for live trading.

        Formula:
            Sharpe = (mean daily return / std daily return) * sqrt(252)
        """
        if returns.std() == 0:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute the maximum drawdown.

        WHY this matters more than total return:
            A strategy that returns +30% but had a -40% drawdown
            at some point is nearly untradeble — most people would
            panic and exit at the bottom, locking in the loss.
            Max drawdown tells you the worst case you'd have lived through.

        Formula:
            For each day, compute how far below the previous peak we are.
            The maximum of those values is the max drawdown.
        """
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())

    @staticmethod
    def _build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a record of every individual trade (entry + exit pair).

        WHY track individual trades:
            Aggregate metrics like Sharpe hide a lot. A strategy might
            have a good Sharpe but 3 huge wins masking many small losses.
            The trade log lets us inspect each trade individually.

        Returns:
            DataFrame with columns:
                entry_date, exit_date, holding_days,
                entry_price, exit_price, trade_return
        """
        trades = []
        in_position = False
        entry_date: Optional[pd.Timestamp] = None
        entry_price: Optional[float] = None

        for date, row in df.iterrows():
            ts = pd.Timestamp(str(date))
            if not in_position and row["position"] == 1:
                in_position = True
                entry_date = ts
                entry_price = float(row["price"])

            elif in_position and row["position"] == 0:
                assert entry_date is not None
                assert entry_price is not None
                exit_price = float(row["price"])
                trade_return = (exit_price / entry_price) - 1
                holding_days = (ts - entry_date).days

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": ts,
                        "holding_days": holding_days,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "trade_return": trade_return,
                    }
                )
                in_position = False
                entry_date = None
                entry_price = None

        if not trades:
            return pd.DataFrame(
                columns=[
                    "entry_date",
                    "exit_date",
                    "holding_days",
                    "entry_price",
                    "exit_price",
                    "trade_return",
                ]
            )

        return pd.DataFrame(trades)

    @staticmethod
    def _win_rate(trades: pd.DataFrame) -> float:
        """
        Fraction of trades with a positive return.

        WHY win rate alone is not enough:
            A strategy can have 40% win rate but still be profitable
            if the average win is 3x the average loss.
            We report win rate alongside avg_holding_days and total
            return so you can see the full picture.
        """
        if len(trades) == 0:
            return 0.0
        return float((trades["trade_return"] > 0).mean())
