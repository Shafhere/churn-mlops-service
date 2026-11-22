
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modeling.data import (
    RAW_DATA,
    RANDOM_STATE,
    make_preprocessor,
    split_data,
)
from modeling.evaluation import (
    FP_COST,
    FN_COST,
    evaluate,
    select_threshold,
)


ARTIFACT_DIR = Path("artifacts")


def get_candidates():
    """Return the models we want to compare."""

    return {
        "dummy_baseline": DummyClassifier(
            strategy="prior"
        ),

        "logistic_regression": LogisticRegression(
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),

        "logistic_balanced": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest_balanced": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main():
    # 1. Read the original dataset.
    df = pd.read_csv(
        RAW_DATA,
        dtype={"customerID": "string"},
    )

    # 2. Validate and split before fitting preprocessing.
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    ) = split_data(df)

    validation_results = []
    trained_models = {}

    # 3. Train every model using the same training split.
    for name, estimator in get_candidates().items():

        pipeline = Pipeline([
            ("preprocessing", make_preprocessor()),
            ("classifier", estimator),
        ])

        pipeline.fit(X_train, y_train)

        # 4. Obtain validation probabilities.
        val_probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        # 5. Find the best validation threshold.
        threshold = select_threshold(
            y_val, val_probabilities
        )

        # 6. Evaluate each candidate.
        metrics = evaluate(
            y_val, val_probabilities, threshold
        )

        result = {
            "model": name,
            **metrics,
        }

        validation_results.append(result)
        trained_models[name] = pipeline

        print(
            f"{name}: "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Cost={metrics['business_cost']}, "
            f"Threshold={threshold:.2f}"
        )

    # 7. Select by validation cost, then PR-AUC.
    winner = min(
        validation_results,
        key=lambda row: (
            row["business_cost"],
            -row["pr_auc"],
            -row["recall"],
            row["model"],
        ),
    )

    best_name = winner["model"]
    best_threshold = winner["threshold"]
    best_model = trained_models[best_name]

    # 8. Evaluate the selected model on test data ONCE.
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        best_threshold,
    )

    # 9. Fingerprint the original training source.
    data_hash = hashlib.sha256(
        RAW_DATA.read_bytes()
    ).hexdigest()

    # 10. Save the model and evaluation information.
    ARTIFACT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        best_model,
        ARTIFACT_DIR / "model.joblib",
    )

    report = {
        "selected_model": best_name,
        "threshold": best_threshold,
        "business_assumptions": {
            "false_positive_cost": FP_COST,
            "false_negative_cost": FN_COST,
        },
        "random_state": RANDOM_STATE,
        "dataset_sha256": data_hash,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "selection_rule": (
            "Minimum validation business cost, "
            "then highest PR-AUC and recall"
        ),
        "validation_comparison": validation_results,
        "test_metrics": test_metrics,
    }

    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n--- SELECTED MODEL ---")
    print("Model:", best_name)
    print("Threshold:", best_threshold)
    print("Test PR-AUC:", test_metrics["pr_auc"])
    print("Test Recall:", test_metrics["recall"])
    print("Test F1:", test_metrics["f1"])
    print("Test Cost:", test_metrics["business_cost"])

    if best_name == "dummy_baseline":
        print(
            "WARNING: Baseline won under the cost "
            "assumption. Investigate before deployment."
        )

    print("\nSaved to:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
