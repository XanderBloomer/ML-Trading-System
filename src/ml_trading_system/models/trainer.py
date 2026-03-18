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
        The threshold can be tuned later for precision/recall tradeoff.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


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
        Train an XGBoost gradient boosted tree classifier.

        WHY XGBoost:
            Decision trees can capture non-linear relationships that
            Logistic Regression cannot. For example:
            "RSI > 70 AND volatility is low → likely reversal"
            This kind of interaction is invisible to logistic regression.

        WHY NO SCALER:
            XGBoost splits on feature values (e.g. "if RSI > 65").
            The actual scale doesn't matter — only the ordering does.
            Adding a scaler would change nothing.

        HYPERPARAMETER DECISIONS:
            n_estimators=300:   Number of trees. Too few = underfitting.
                                Too many = overfitting. 300 is a safe start.
            max_depth=4:        Maximum tree depth. Shallow trees (3-5)
                                generalise better on financial data which
                                is mostly noise. Deep trees memorise noise.
            learning_rate=0.05: How much each tree corrects the previous.
                                Lower = more trees needed but better
                                generalisation. 0.05 is conservative.
            subsample=0.8:      Each tree sees 80% of training rows
                                (random). Reduces overfitting.
            colsample_bytree=0.8: Each tree sees 80% of features.
                                  Reduces overfitting further.
            eval_metric="logloss": We care about predicted probabilities,
                                   not just class labels. Log loss
                                   measures probability calibration.
            random_state=42:    Reproducibility.

        NOTE: These are starting hyperparameters. We will tune them
        properly using walk-forward cross-validation in a later phase.
        """
        # --- Decision: compute scale_pos_weight ---
        # XGBoost's equivalent of class_weight="balanced".
        # scale_pos_weight = count(negative class) / count(positive class)
        # If 530 up days and 470 down days:
        #   scale_pos_weight = 470 / 530 = 0.887
        # This tells XGBoost to weight the minority class higher.
        n_down = (split.y_train == 0).sum()
        n_up = (split.y_train == 1).sum()
        scale_pos_weight = n_down / n_up if n_up > 0 else 1.0

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,  # suppress XGBoost's own output
        )

        model.fit(split.X_train, split.y_train)

        # --- Store feature importances in metadata ---
        # XGBoost gives us feature importances for free.
        # We store them now so the evaluator can display them
        # without needing access to the raw model internals.
        importances = dict(
            zip(
                split.X_train.columns,
                model.feature_importances_,
            )
        )

        return TrainedModel(
            name="xgboost",
            model=model,
            feature_names=list(split.X_train.columns),
            train_start=split.train_start,
            train_end=split.train_end,
            metadata={"feature_importances": importances},
        )
