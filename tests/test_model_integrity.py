"""Guard against accidental or untrusted model artifact replacement."""

import hashlib
import json

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from api.service import ModelService


def test_tampered_model_file_rejected(tmp_path):
    model = DummyClassifier(strategy="prior").fit(
        pd.DataFrame({"x": [1, 2]}), [0, 1]
    )
    path = tmp_path / "model_v1.joblib"
    joblib.dump(model, path)
    metadata = {
        "registry": {"version": "1"},
        "threshold": 0.37,
        "selected_model": "dummy_baseline",
        "test_metrics": {},
        "model_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    with path.open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ModelService.load(tmp_path)
