
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Hypothetical business costs.
FP_COST = 1
FN_COST = 5


def threshold_metrics(y_true, probabilities, threshold):
    """Calculate predictions and business cost."""

    predicted = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(
            float(accuracy_score(y_true, predicted)), 4
        ),
        "precision": round(
            float(precision_score(
                y_true, predicted, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_true, predicted, zero_division=0
            )), 4
        ),
        "f1": round(
            float(f1_score(
                y_true, predicted, zero_division=0
            )), 4
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": int(
            fp * FP_COST + fn * FN_COST
        ),
    }


def select_threshold(y_true, probabilities):
    """Choose the lowest-cost threshold on validation data."""

    candidates = [
        threshold_metrics(
            y_true, probabilities, i / 100
        )
        for i in range(1, 100)
    ]

    best = min(
        candidates,
        key=lambda row: (
            row["business_cost"],
            -row["recall"],
            abs(row["threshold"] - 0.5),
        ),
    )

    return best["threshold"]


def evaluate(y_true, probabilities, threshold):
    """Calculate threshold and probability-based metrics."""

    metrics = threshold_metrics(
        y_true, probabilities, threshold
    )

    # These assess probability ranking, not only labels.
    metrics["pr_auc"] = round(
        float(average_precision_score(
            y_true, probabilities
        )), 4
    )

    metrics["roc_auc"] = round(
        float(roc_auc_score(
            y_true, probabilities
        )), 4
    )

    # Lower Brier score is better.
    metrics["brier_score"] = round(
        float(brier_score_loss(
            y_true, probabilities
        )), 4
    )

    return metrics
