"""
Time-based train/test splitting for financial time series.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitResult:
    """
    Holds the output of a train/test split.

    Attributes:
        X_train: Feature matrix for training.
        X_test:  Feature matrix for testing (out-of-sample).
        y_train: Target vector for training.
        y_test:  Target vector for testing.
        train_start: First date in training set.
        train_end:   Last date in training set.
        test_start:  First date in test set.
        test_end:    Last date in test set.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> str:
        return (
            f"Train: {self.train_start.date()} → {self.train_end.date()} "
            f"({len(self.X_train)} rows)\n"
            f"Test:  {self.test_start.date()} → {self.test_end.date()} "
            f"({len(self.X_test)} rows)"
        )


class TimeSeriesSplitter:
    """
    Splits a feature DataFrame into train and test sets by date.

    Args:
        test_size: Fraction of data to use as test set (default: 0.2 = 20%).
        gap_days:  Number of calendar days to drop between train and test.
                   Simulates the delay between training a model and deploying
                   it live. Prevents subtle boundary leakage. Default: 30.
    """

    def __init__(self, test_size: float = 0.2, gap_days: int = 30) -> None:
        # Decision: test_size=0.2
        # 80% train / 20% test is the standard starting point.
        # For our 2020-2024 dataset (1006 rows after feature warmup):
        #   ~800 rows training  (~2020-01-01 to ~2023-03-01)
        #   ~200 rows testing   (~2023-04-01 to ~2024-01-01)
        # The test set covers a full 9 months — enough to evaluate properly.
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if gap_days < 0:
            raise ValueError(f"gap_days must be >= 0, got {gap_days}")

        self.test_size = test_size
        self.gap_days = gap_days

    def split(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        target_col: str = "target",
    ) -> SplitResult:
        """
        Split df into train and test sets by date.

        Args:
            df:            DataFrame with a DatetimeIndex, features, and target.
            feature_names: List of column names to use as features (X).
            target_col:    Name of the target column (y). Default: "target".

        Returns:
            SplitResult with X_train, X_test, y_train, y_test and date metadata.

        Raises:
            ValueError: If target_col or any feature_name is missing from df.
            ValueError: If df has fewer than 50 rows (not enough to split).
        """
        # Validation
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found in DataFrame: {missing}")
        if len(df) < 50:
            raise ValueError(f"DataFrame too small to split: {len(df)} rows.")

        # Sort by index
        df = df.sort_index()

        # Compute split date
        n = len(df)
        train_end_idx = int(n * (1 - self.test_size))
        split_date = df.index[train_end_idx]

        # Apply gap
        test_start_date = split_date + pd.Timedelta(days=self.gap_days)

        train_df = df[df.index < split_date]
        test_df = df[df.index >= test_start_date]

        if len(train_df) == 0:
            raise ValueError("Train set is empty after split. Adjust test_size.")
        if len(test_df) == 0:
            raise ValueError(
                "Test set is empty after split. Adjust test_size or gap_days."
            )

        # Extract X and y
        # X = features only (never includes target)
        # y = target only (never includes features)
        X_train = train_df[feature_names]
        X_test = test_df[feature_names]
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
        )
