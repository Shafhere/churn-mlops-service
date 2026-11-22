
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient

from modeling.data import RAW_DATA, split_data
from modeling.train import (
    TRACKING_URI,
    file_sha256,
)


def main():
    metadata = json.loads(
        Path("artifacts/model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    # Confirm the source dataset has not changed.
    assert file_sha256(RAW_DATA) == (
        metadata["dataset"]["source_sha256"]
    ), "Dataset hash mismatch"

    # Load the exact locally saved version.
    local_model = joblib.load(
        metadata["model_file"]
    )

    # Load the exact registered MLflow version.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    registry = metadata["registry"]

    registered_model = mlflow.sklearn.load_model(
        registry["uri"]
    )

    version = MlflowClient().get_model_version(
        registry["name"],
        registry["version"],
    )

    assert (
        version.tags["source_sha256"]
        == metadata["dataset"]["source_sha256"]
    )

    # Both copies must produce matching probabilities.
    df = pd.read_csv(
        RAW_DATA,
        dtype={"customerID": "string"},
    )

    _, _, X_test, _, _, _ = split_data(df)
    sample = X_test.head(5)

    local_probs = local_model.predict_proba(
        sample
    )[:, 1]

    registry_probs = registered_model.predict_proba(
        sample
    )[:, 1]

    assert np.allclose(
        local_probs, registry_probs
    ), "Saved and registered models disagree"

    print("PASS: Dataset fingerprint matches")
    print("PASS: Registry version exists")
    print("PASS: Both models load successfully")
    print("PASS: Prediction probabilities agree")
    print("Model URI:", registry["uri"])
    print("Threshold:", metadata["threshold"])


if __name__ == "__main__":
    main()
