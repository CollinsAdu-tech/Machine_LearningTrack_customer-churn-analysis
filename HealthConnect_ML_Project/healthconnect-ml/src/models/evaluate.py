"""
HealthConnect ML Pipeline — Model Evaluation
Week 5 — Machine Learning Engineering Track

Computes the metrics specified in the Data Science handoff, Section 10:
    Primary:   Recall (No-Show / positive class)
    Secondary: Precision, F1-score, Confusion Matrix, ROC-AUC, Accuracy

Recall is reported first and explicitly labelled as primary, since a false
negative (predicting Attended when the patient actually no-shows) is the
specific failure the project exists to reduce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src import config


@dataclass
class EvaluationResult:
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    accuracy: float = 0.0
    confusion_matrix: list = field(default_factory=list)

    def summary(self) -> str:
        tn, fp, fn, tp = np.array(self.confusion_matrix).ravel()
        lines = [
            "HealthConnect Baseline Model Evaluation",
            "=" * 40,
            f"Recall (No-Show)   [PRIMARY]: {self.recall:.3f}",
            f"Precision (No-Show)          : {self.precision:.3f}",
            f"F1-score                     : {self.f1:.3f}",
            f"ROC-AUC                      : {self.roc_auc:.3f}",
            f"Accuracy                     : {self.accuracy:.3f}",
            "",
            "Confusion Matrix (rows=actual, cols=predicted; order [Attended, No-Show]):",
            f"  TN={tn}  FP={fp}",
            f"  FN={fn}  TP={tp}",
            "",
            f"False negatives (missed no-shows): {fn} — this is the primary error to minimize.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> EvaluationResult:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    result = EvaluationResult(
        recall=float(recall_score(y_test, y_pred, pos_label=1)),
        precision=float(precision_score(y_test, y_pred, pos_label=1)),
        f1=float(f1_score(y_test, y_pred, pos_label=1)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    )
    return result


if __name__ == "__main__":
    from src.models.train import train_baseline_model

    fitted_pipeline, _, (X_test, y_test) = train_baseline_model()
    eval_result = evaluate_model(fitted_pipeline, X_test, y_test)
    print(eval_result.summary())

    # Persist results for the reports/ directory
    results_path = config.PROJECT_ROOT / "reports" / "model_results" / "baseline_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(eval_result.to_dict(), f, indent=2)
