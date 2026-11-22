import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.schemas import CustomerInput, Factor, PredictionOutput

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelService:
    model: Any
    model_name: str
    version: str
    threshold: float
    test_metrics: dict
    reference: dict | None = None

    @classmethod
    def load(cls, artifact_dir: Path | None = None):
        artifact_dir = artifact_dir or ROOT / "artifacts"

        metadata = json.loads(
            (artifact_dir / "model_metadata.json").read_text(
                encoding="utf-8"
            )
        )

        version = str(metadata["registry"]["version"])

        if not (version.isascii() and version.isdigit()):
            raise ValueError("Invalid model version")

        threshold = float(metadata["threshold"])

        if not 0 < threshold < 1:
            raise ValueError("Invalid threshold")

        model = joblib.load(
            artifact_dir / f"model_v{version}.joblib"
        )

        if not callable(
            getattr(model, "predict_proba", None)
        ):
            raise TypeError("Model must support predict_proba")

        reference_file = artifact_dir / "monitor_reference.json"

        reference = (
            json.loads(reference_file.read_text(encoding="utf-8"))
            if reference_file.exists()
            else None
        )

        # Prevent mixing a model with reference data
        # from a different source dataset.
        if reference and (
            reference["source_sha256"]
            != metadata["dataset"]["source_sha256"]
        ):
            raise ValueError(
                "Monitoring reference dataset mismatch"
            )

        return cls(
            model=model,
            model_name=metadata["selected_model"],
            version=version,
            threshold=threshold,
            test_metrics=metadata["test_metrics"],
            reference=reference,
        )

    def predict(
        self, records: list[CustomerInput]
    ) -> list[PredictionOutput]:
        """Run batch inference using the saved pipeline."""

        frame = pd.DataFrame([
            record.model_dump() for record in records
        ])

        probabilities = np.asarray(
            self.model.predict_proba(frame)[:, 1]
        )

        if not np.isfinite(probabilities).all():
            raise ValueError("Invalid model probabilities")

        return [
            PredictionOutput(
                churn_probability=round(float(p), 4),
                prediction=(
                    "Yes" if p >= self.threshold else "No"
                ),
                threshold=self.threshold,
                model_version=self.version,
            )
            for p in probabilities
        ]

    def explain(self, customer: CustomerInput) -> list[Factor]:
        """Estimate local sensitivity to individual features."""

        baseline = (self.reference or {}).get("baseline", {})

        if not baseline:
            return []

        original = customer.model_dump()
        rows = [original]
        feature_names = []

        for name, typical_value in baseline.items():
            if name not in original:
                continue

            if original[name] == typical_value:
                continue

            changed = original.copy()
            changed[name] = typical_value

            rows.append(changed)
            feature_names.append(name)

        scores = self.model.predict_proba(
            pd.DataFrame(rows)
        )[:, 1]

        explanations = []

        for name, modified_score in zip(
            feature_names, scores[1:]
        ):
            impact = float(scores[0] - modified_score)

            if abs(impact) < 0.0001:
                continue

            explanations.append(
                Factor(
                    feature=name,
                    probability_impact=round(impact, 4),
                    direction=(
                        "increases risk"
                        if impact > 0 else "decreases risk"
                    ),
                )
            )

        explanations.sort(
            key=lambda item: abs(item.probability_impact),
            reverse=True,
        )

        return explanations[:3]