"""
Model evaluation for trading signal quality.

WHY STANDARD ML METRICS ARE NOT ENOUGH:
    Accuracy alone is misleading in finance.
    A model that predicts "up" every day scores ~53% accuracy
    on US equities — without learning anything useful.

    We need metrics that answer trading-relevant questions:
    - When the model says "buy", how often is it right? (Precision)
    - How confident are the predictions? (Log loss / Brier score)
    - What is the information coefficient? (Rank correlation)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from ml_trading_system.models.trainer import TrainedModel


@dataclass
class EvaluationResult:
    """
    Holds all evaluation metrics for a single model on a single dataset.

    Attributes:
        model_name:           Name of the model evaluated.
        accuracy:             Fraction of correct directional predictions.
        directional_accuracy: Same as accuracy but named explicitly for clarity.
        precision:            Of all "up" predictions, fraction that were correct.
                              This is the most trading-relevant metric.
        recall:               Of all actual "up" days, fraction the model caught.
        log_loss:             Measures probability calibration (lower = better).
                              A perfectly calibrated model scores ~0.693 (random).
                              Below 0.693 = model has learned something.
        brier_score:          Mean squared error of predicted probabilities.
                              Lower = better. Random = 0.25.
        n_samples:            Number of test samples evaluated.
        up_ratio:             Fraction of test days that were actually "up".
                              Used to detect if target is imbalanced.
    """

    model_name: str
    accuracy: float
    directional_accuracy: float
    precision: float
    recall: float
    log_loss: float
    brier_score: float
    n_samples: int
    up_ratio: float

    def summary(self) -> str:
        """Print a clean summary of all metrics."""
        # Decision: show random baseline next to each metric
        # This prevents misreading results. 53% accuracy sounds good
        # until you see the random baseline is also 53%.
        random_acc = self.up_ratio  # predicting "up" always = up_ratio accuracy
        return (
            f"\n{'='*50}\n"
            f"Model: {self.model_name}\n"
            f"{'='*50}\n"
            f"Samples evaluated : {self.n_samples}\n"
            f"Market up ratio   : {self.up_ratio:.1%}  ← random baseline\n"
            f"\n--- Direction ---\n"
            f"Accuracy          : {self.accuracy:.1%}  (random: {random_acc:.1%})\n"
            f"Directional acc.  : {self.directional_accuracy:.1%}\n"
            f"Precision (up)    : {self.precision:.1%}\n"
            f"Recall (up)       : {self.recall:.1%}\n"
            f"\n--- Probability Quality ---\n"
            f"Log loss          : {self.log_loss:.4f}  (random: 0.6931)\n"
            f"Brier score       : {self.brier_score:.4f}  (random: 0.2500)\n"
            f"{'='*50}\n"
        )


class ModelEvaluator:
    """
    Evaluates a TrainedModel on out-of-sample test data.

    Usage:
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X_test, y_test)
        print(result.summary())
    """

    def evaluate(
        self,
        trained_model: TrainedModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics for a trained model.

        Args:
            trained_model: A TrainedModel from ModelTrainer.
            X_test:        Feature matrix (out-of-sample).
            y_test:        True labels (out-of-sample).

        Returns:
            EvaluationResult with all metrics populated.
        """
        # Get predictions
        # predict_proba gives us probability of each class.
        # proba[:, 1] = P(up) for each day.
        # We use this for probability metrics (log loss, brier).
        # For direction metrics we threshold at 0.5.
        proba = trained_model.predict_proba(X_test)
        preds = trained_model.predict(X_test)
        proba_up = proba[:, 1]

        # Decision: zero_division=0 in precision/recall
        # If the model never predicts "up" (degenerate edge case),
        # sklearn raises a warning and returns 0. We handle this
        # explicitly so it doesn't pollute output.
        return EvaluationResult(
            model_name=trained_model.name,
            accuracy=float(accuracy_score(y_test, preds)),
            directional_accuracy=float(accuracy_score(y_test, preds)),
            precision=float(precision_score(y_test, preds, zero_division=0)),
            recall=float(recall_score(y_test, preds, zero_division=0)),
            log_loss=float(log_loss(y_test, proba_up)),
            brier_score=float(brier_score_loss(y_test, proba_up)),
            n_samples=int(len(y_test)),
            up_ratio=float(y_test.mean()),
        )

    def compare(
        self,
        results: list[EvaluationResult],
    ) -> pd.DataFrame:
        """
        Compare multiple EvaluationResults side by side as a DataFrame.

        Useful for printing a clean comparison table of all models.

        Args:
            results: List of EvaluationResult objects to compare.

        Returns:
            DataFrame with one row per model, columns = metrics.
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "model": r.model_name,
                    "accuracy": f"{r.accuracy:.1%}",
                    "precision": f"{r.precision:.1%}",
                    "recall": f"{r.recall:.1%}",
                    "log_loss": f"{r.log_loss:.4f}",
                    "brier_score": f"{r.brier_score:.4f}",
                    "n_samples": r.n_samples,
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def feature_importance(
        self,
        trained_model: TrainedModel,
        top_n: int = 10,
    ) -> pd.Series:
        """
        Return feature importances for XGBoost models.

        For Logistic Regression, returns the absolute coefficient values
        (which act as a proxy for importance after StandardScaler).

        Args:
            trained_model: A TrainedModel object.
            top_n:         Number of top features to return.

        Returns:
            pd.Series sorted by importance descending.
        """
        # XGBoost: importances stored in metadata during training
        if "feature_importances" in trained_model.metadata:
            importances = trained_model.metadata["feature_importances"]
            series = pd.Series(importances).sort_values(ascending=False)
            return series.head(top_n)

        # Logistic Regression: use absolute coefficients
        # After StandardScaler, all features are on the same scale.
        # The absolute value of the coefficient tells us how strongly
        # each feature influences the prediction direction.
        if not isinstance(trained_model.model, Pipeline):
            raise TypeError(
                f"Cannot extract coefficients from model type: "
                f"{type(trained_model.model).__name__}. "
                f"Expected sklearn Pipeline."
            )
        coefficients = trained_model.model.named_steps["model"].coef_[0]
        series = pd.Series(
            np.abs(coefficients),
            index=trained_model.feature_names,
        ).sort_values(ascending=False)
        return series.head(top_n)
