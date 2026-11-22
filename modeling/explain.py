
from sklearn.inspection import permutation_importance


def get_feature_importance(model, X_val, y_val):
    """Measure global importance using validation data."""

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    importance = [
        {
            "feature": feature,
            "importance": round(float(score), 6),
        }
        for feature, score in zip(
            X_val.columns,
            result.importances_mean,
        )
    ]

    # Higher importance first.
    importance.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return importance[:10]
