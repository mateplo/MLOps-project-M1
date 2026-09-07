"""Metric helpers shared by train / evaluate / promote."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold(y_true, y_score) -> tuple[float, float]:
    """Decision threshold maximising F1 on (y_true, y_score). Returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns n+1 points for n thresholds; drop the last one.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def compute_metrics(y_true, y_score, threshold: float = 0.5, prefix: str = "test") -> dict:
    """Threshold-free metrics + threshold-dependent metrics at `threshold` (and at 0.5)."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_pred_default = (np.asarray(y_score) >= 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_score)),
        f"{prefix}_average_precision": float(average_precision_score(y_true, y_score)),
        f"{prefix}_brier": float(brier_score_loss(y_true, y_score)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred)),
        f"{prefix}_f1_at_0.5": float(f1_score(y_true, y_pred_default)),
        "decision_threshold": float(threshold),
    }
