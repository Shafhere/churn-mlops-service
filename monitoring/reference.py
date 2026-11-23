"""Build monitoring statistics using training data only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.data import FEATURES, NUMERIC, RAW_DATA, split_data

OUTPUT = Path("artifacts/monitor_reference.json")

WATCH = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
]

MISSING = "__MISSING__"
OTHER = "__OTHER__"


def bucket(value, spec):
    """Assign a value to a monitoring category."""

    if pd.isna(value):
        return MISSING

    if spec["type"] == "numeric":
        try:
            number = float(value)

            if not np.isfinite(number):
                return MISSING

            index = np.searchsorted(
                spec["edges"], number, side="right"
            )
            return f"bin_{int(index)}"

        except (TypeError, ValueError):
            return OTHER

    value = str(value)
    return value if value in spec["counts"] else OTHER


def build_reference(X_train):
    """Store aggregate distributions and typical values."""

    baseline = {}

    for feature in FEATURES:
        if feature in NUMERIC:
            values = pd.to_numeric(
                X_train[feature], errors="coerce"
            ).dropna()

            baseline[feature] = float(values.median())
        else:
            values = X_train[feature].dropna()
            baseline[feature] = str(values.mode().iloc[0])

    features = {}

    for name in WATCH:
        values = X_train[name]

        if name in NUMERIC:
            numeric = pd.to_numeric(
                values, errors="coerce"
            ).dropna()

            edges = sorted({
                float(x)
                for x in np.quantile(
                    numeric, [0.2, 0.4, 0.6, 0.8]
                )
            })

            counts = {
                f"bin_{i}": 0
                for i in range(len(edges) + 1)
            }
            counts.update({MISSING: 0, OTHER: 0})

            spec = {
                "type": "numeric",
                "edges": edges,
                "counts": counts,
            }

        else:
            categories = sorted(
                values.dropna().astype(str).unique().tolist()
            )
            counts = {category: 0 for category in categories}
            counts.update({MISSING: 0, OTHER: 0})

            spec = {
                "type": "categorical",
                "counts": counts,
            }

        for value in values:
            label = bucket(value, spec)
            spec["counts"][label] += 1

        features[name] = spec

    return {"baseline": baseline, "features": features}


def main():
    df = pd.read_csv(
        RAW_DATA, dtype={"customerID": "string"}
    )

    X_train, _, _, _, _, _ = split_data(df)

    reference = build_reference(X_train)

    reference["source_sha256"] = hashlib.sha256(
        RAW_DATA.read_bytes()
    ).hexdigest()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print("Reference created:", OUTPUT)
    print("Monitored features:", ", ".join(reference["features"]))


if __name__ == "__main__":
    main()