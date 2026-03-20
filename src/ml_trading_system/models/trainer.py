"""
Model training for direction prediction.

WHY TWO MODELS:
    Logistic Regression = baseline. Simple, fast, interpretable.
    XGBoost = our first "real" model. Handles non-linear relationships.

    Always establish a baseline before adding complexity.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml_trading_system.models.splitter import SplitResult


class ModelProtocol(Protocol):
    """
    Defines the interface every model must satisfy.

    WHY A PROTOCOL:
        Python Protocols define a structural interface without inheritance.
        Any object with fit() and predict_proba() methods satisfies this.
        This means we can swap models freely without changing the trainer.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class TrainedModel:
    """
    Wraps a trained model with its metadata.

    WHY WRAP THE MODEL:
        A raw sklearn model has no memory of what it was trained on.
        By wrapping it, we always know:
        - which features it expects (prevents silent wrong-column bugs)
        - what date range it was trained on
        - what model type it is
        This makes the system much safer when serving predictions later.

    Attributes:
        name:          Human-readable model name (e.g. "logistic_regression").
        model:         The fitted sklearn-compatible model object.
        feature_names: Ordered list of feature columns the model expects.
        train_start:   First date in the training data.
        train_end:     Last date in the training data.
    """

    name: str
    model: ModelProtocol
    feature_names: list[str]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Enforces that X has exactly the right feature columns
        in the right order. Prevents silent bugs from column mismatches.

        Returns:
            Array of shape (n_samples, 2).
            Column 0 = P(down), Column 1 = P(up).
        """
        # Decision: enforce feature alignment
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        return self.model.predict_proba(X[self.feature_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (0 = down, 1 = up).
        Uses predict_proba with threshold=0.5.
        """
        return self.predict_with_threshold(X, threshold=0.5)

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels using a custom probability threshold.

        WHY THIS EXISTS:
            Default predict() uses 0.5 — trade whenever P(up) > 50%.
            But a 51% confident signal is barely better than a coin flip.
            By raising the threshold to e.g. 0.6, we only trade when
            the model is genuinely confident. This means:
              - Fewer trades (less commission drag)
              - Higher quality trades (higher precision)
              - More days in cash (missing some gains, avoiding some losses)

            The trade-off: higher threshold = fewer signals = smaller
            sample size for the backtest. Too high and you barely trade.
            0.55 and 0.60 are sensible values to test first.

        Args:
            X:         Feature matrix.
            threshold: Minimum P(up) required to generate a buy signal.
                       Must be between 0.5 and 1.0.
                       Below 0.5 would invert the model's logic.

        Returns:
            Array of 0s and 1s. 1 = P(up) >= threshold, 0 = hold cash.
        """
        if not 0.5 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.5 and 1.0, got {threshold}. "
                f"Values below 0.5 would invert the model's direction."
            )
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


class ModelTrainer:
    """
    Trains Logistic Regression and XGBoost models on a SplitResult.

    Usage:
        trainer = ModelTrainer()
        models = trainer.train_all(split)
        lr_model = models["logistic_regression"]
        xgb_model = models["xgboost"]
    """

    def train_all(self, split: SplitResult) -> dict[str, TrainedModel]:
        """
        Train all models and return them as a dict keyed by name.

        Args:
            split: A SplitResult from TimeSeriesSplitter.split().

        Returns:
            Dict mapping model name → TrainedModel.
        """
        return {
            "logistic_regression": self._train_logistic_regression(split),
            "xgboost": self._train_xgboost(split),
        }

    def _train_logistic_regression(self, split: SplitResult) -> TrainedModel:
        """
        Train a Logistic Regression classifier.

        WHY A PIPELINE (StandardScaler + LogisticRegression):
            Logistic Regression uses gradient descent internally.
            Features on different scales cause slow convergence and
            poor weight estimates. StandardScaler transforms each
            feature to mean=0, std=1 before fitting.

            Crucially, the scaler is fit ONLY on training data.
            When we call predict on test data, the scaler applies
            the training mean/std — not the test mean/std.
            This is correct. Fitting the scaler on test data would
            be another form of data leakage.

        WHY class_weight="balanced":
            Markets go up ~53% of days. Without this, the model
            could predict "up" every day and score 53% accuracy
            while learning nothing. "balanced" weights each class
            inversely proportional to its frequency, forcing the
            model to actually learn both directions.

        WHY max_iter=1000:
            Default is 100, which often fails to converge on
            financial data. 1000 is safe.
        """
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(split.X_train, split.y_train)

        return TrainedModel(
            name="logistic_regression",
            model=pipeline,
            feature_names=list(split.X_train.columns),
            train_start=split.train_start,
            train_end=split.train_end,
        )

    def _train_xgboost(self, split: SplitResult) -> TrainedModel:
        """
        Train an XGBoost classifier with isotonic probability calibration.

        WHY CalibratedClassifierCV:
            XGBoost optimises for ranking not probability accuracy.
            Raw probabilities are unreliable — 0.6 confidence may only
            be right 52% of the time. Isotonic calibration fits a
            non-parametric correction layer on top using 5-fold CV
            on training data only. After calibration, P(up)=0.6 means
            the market went up ~60% of the time at that confidence level.

        WHY method="isotonic" not "sigmoid":
            Sigmoid assumes symmetric monotonic miscalibration.
            Isotonic is non-parametric — no shape assumptions.
            Financial data is rarely symmetric so isotonic is safer.
        """
        n_down = (split.y_train == 0).sum()
        n_up = (split.y_train == 1).sum()
        scale_pos_weight = n_down / n_up if n_up > 0 else 1.0

        base_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        # Wrap in calibration layer — cv=5 uses training data only
        model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=5,
        )

        model.fit(split.X_train, split.y_train)

        # Average feature importances across the 5 calibrated folds
        importances_per_fold = np.array(
            [
                est.estimator.feature_importances_
                for est in model.calibrated_classifiers_
            ]
        )
        mean_importances = importances_per_fold.mean(axis=0)
        importances = dict(zip(split.X_train.columns, mean_importances))

        return TrainedModel(
            name="xgboost",
            model=model,
            feature_names=list(split.X_train.columns),
            train_start=split.train_start,
            train_end=split.train_end,
            metadata={"feature_importances": importances},
        )
