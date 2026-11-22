
import hashlib
import json
import platform
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modeling.data import (
    RAW_DATA,
    RANDOM_STATE,
    split_data,
    make_preprocessor,
)
from modeling.evaluation import (
    FP_COST,
    FN_COST,
    evaluate,
    select_threshold,
)
from modeling.explain import get_feature_importance
from modeling.model_card import build_model_card


ARTIFACT_DIR = Path("artifacts")
CARD_PATH = Path("docs/MODEL_CARD.md")

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "ChurnGuard-Experiments"
REGISTERED_MODEL = "churnguard-churn-model"


def file_sha256(path):
    """Fingerprint the exact original file bytes."""
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def training_sha256(X_train, y_train):
    """Fingerprint the selected training rows and labels."""
    training = X_train.copy()
    training["__target__"] = y_train

    content = training.to_csv(
        index=True
    ).encode("utf-8")

    return hashlib.sha256(content).hexdigest()


def get_candidates():
    """Candidate models with reproducible settings."""
    return {
        "dummy_baseline": DummyClassifier(
            strategy="prior",
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
    # MLflow uses one local database for experiments and
    # registered model metadata.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(
        RAW_DATA,
        dtype={"customerID": "string"},
    )

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    ) = split_data(df)

    source_hash = file_sha256(RAW_DATA)
    train_hash = training_sha256(X_train, y_train)

    results = []
    trained = {}

    print("\nTraining and tracking candidate models...\n")

    # STEP 1: Compare models on the validation split.
    for name, estimator in get_candidates().items():

        with mlflow.start_run(run_name=name) as run:

            pipeline = Pipeline([
                ("preprocessing", make_preprocessor()),
                ("classifier", estimator),
            ])

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_val
            )[:, 1]

            threshold = select_threshold(
                y_val, probabilities
            )

            metrics = evaluate(
                y_val, probabilities, threshold
            )

            # Track training and business assumptions.
            mlflow.log_params({
                "model_name": name,
                "random_state": RANDOM_STATE,
                "fp_cost": FP_COST,
                "fn_cost": FN_COST,
                "split": "70/15/15",
            })

            mlflow.log_params({
                f"est_{key}": str(value)
                for key, value in
                estimator.get_params(deep=False).items()
            })

            mlflow.log_metrics({
                f"val_{key}": float(value)
                for key, value in metrics.items()
            })

            mlflow.set_tags({
                "source_sha256": source_hash,
                "train_split_sha256": train_hash,
                "evaluation": "validation",
            })

            results.append({
                "model": name,
                "run_id": run.info.run_id,
                **metrics,
            })

            trained[name] = pipeline

            print(
                f"{name} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Cost={metrics['business_cost']} | "
                f"Threshold={threshold:.2f}"
            )

    # STEP 2: Choose using validation results only.
    winner = min(
        results,
        key=lambda row: (
            row["business_cost"],
            -row["pr_auc"],
            -row["recall"],
            row["model"],
        ),
    )

    best_name = winner["model"]
    best_model = trained[best_name]
    best_threshold = winner["threshold"]

    # STEP 3: Final untouched holdout evaluation.
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        best_threshold,
    )

    # STEP 4: Global feature importance on validation.
    importance = get_feature_importance(
        best_model, X_val, y_val
    )

    ARTIFACT_DIR.mkdir(
        parents=True, exist_ok=True
    )
    CARD_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    # STEP 5: Log and register the selected model.
    with mlflow.start_run(
        run_name=f"selected_{best_name}"
    ) as run:

        mlflow.log_params({
            "selected_model": best_name,
            "threshold": best_threshold,
            "random_state": RANDOM_STATE,
            "fp_cost": FP_COST,
            "fn_cost": FN_COST,
        })

        mlflow.log_metrics({
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
        })

        mlflow.set_tags({
            "source_sha256": source_hash,
            "train_split_sha256": train_hash,
            "deployment_status": "candidate",
        })

        # Signature stores feature names and types,
        # without saving customer records as input examples.
        sample = X_train.head(3)
        signature = infer_signature(
            sample,
            best_model.predict(sample),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="churn_pipeline",
            signature=signature,
            serialization_format="cloudpickle",
        )

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL,
        )

        version = str(registered.version)
        registry_uri = (
            f"models:/{REGISTERED_MODEL}/{version}"
        )

        # Tag the registered version for traceability.
        client = MlflowClient()

        for key, value in {
            "approval_status": "candidate",
            "decision_threshold": best_threshold,
            "source_sha256": source_hash,
            "train_split_sha256": train_hash,
        }.items():
            client.set_model_version_tag(
                REGISTERED_MODEL,
                version,
                key,
                str(value),
            )

        # Keep a local versioned copy for later API serving.
        model_file = (
            ARTIFACT_DIR / f"model_v{version}.joblib"
        )

        joblib.dump(best_model, model_file)

        # STEP 6: Complete versioned metadata.
        report = {
            "status": "candidate_needs_approval",
            "selected_model": best_name,
            "threshold": best_threshold,
            "random_state": RANDOM_STATE,
            "business_assumptions": {
                "false_positive_cost": FP_COST,
                "false_negative_cost": FN_COST,
            },
            "dataset": {
                "records": len(df),
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "test_rows": len(y_test),
                "source_sha256": source_hash,
                "train_split_sha256": train_hash,
            },
            "registry": {
                "name": REGISTERED_MODEL,
                "version": version,
                "uri": registry_uri,
                "run_id": run.info.run_id,
            },
            "model_file": str(model_file.as_posix()),
            "validation_comparison": results,
            "test_metrics": test_metrics,
            "feature_importance": importance,
            "runtime": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "mlflow": mlflow.__version__,
            },
        }

        metadata_file = (
            ARTIFACT_DIR / "model_metadata.json"
        )

        metadata_file.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # STEP 7: Automatically create the model card.
        CARD_PATH.write_text(
            build_model_card(report),
            encoding="utf-8",
        )

        # Store documentation with the MLflow run.
        mlflow.log_dict(
            report, "model_metadata.json"
        )
        mlflow.log_artifact(str(CARD_PATH))

    print("\n===== FINAL MODEL =====")
    print("Selected:", best_name)
    print("Registry version:", version)
    print("Decision threshold:", best_threshold)
    print("Test PR-AUC:", test_metrics["pr_auc"])
    print("Test Recall:", test_metrics["recall"])
    print("Test F1:", test_metrics["f1"])
    print("Test business cost:", test_metrics["business_cost"])
    print("Model:", model_file)
    print("Metadata:", metadata_file)
    print("Model card:", CARD_PATH)

    if best_name == "dummy_baseline":
        print(
            "WARNING: Baseline selected. Investigate "
            "before approving any deployment."
        )


if __name__ == "__main__":
    main()
