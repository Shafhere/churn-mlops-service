
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api.main import app
from api.schemas import CustomerInput, PredictionOutput
from api.service import ModelService

KEY = "test-only-api-key"
HEADERS = {"X-API-Key": KEY}

SAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 45.0,
    "TotalCharges": 540.0,
}


class FakeService:
    """Fake predictions for testing API behavior."""

    version = "test"
    model_name = "test_model"
    threshold = 0.4
    test_metrics = {"recall": 0.8}

    def predict(self, records):
        return [
            PredictionOutput(
                churn_probability=0.7,
                prediction="Yes",
                threshold=self.threshold,
                model_version=self.version,
            )
            for _ in records
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_API_KEY", KEY)

    monkeypatch.setattr(
        ModelService,
        "load",
        classmethod(lambda cls: FakeService()),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_missing_key_is_rejected(client):
    response = client.post(
        "/predict", json=SAMPLE
    )
    assert response.status_code == 401


def test_wrong_key_is_rejected(client):
    response = client.post(
        "/predict",
        json=SAMPLE,
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_single_prediction(client):
    response = client.post(
        "/predict",
        json=SAMPLE,
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "Yes"
    assert response.json()["threshold"] == 0.4


def test_invalid_input_and_no_sensitive_echo(client):
    invalid = {
        **SAMPLE,
        "MonthlyCharges": -99,
    }

    response = client.post(
        "/predict",
        json=invalid,
        headers=HEADERS,
    )

    assert response.status_code == 422
    assert "-99" not in json.dumps(response.json())


def test_extra_field_rejected(client):
    response = client.post(
        "/predict",
        json={**SAMPLE, "customerID": "private"},
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_batch_and_limit(client):
    response = client.post(
        "/predict/batch",
        json={"records": [SAMPLE, SAMPLE]},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["count"] == 2

    empty = client.post(
        "/predict/batch",
        json={"records": []},
        headers=HEADERS,
    )
    assert empty.status_code == 422

    too_large = client.post(
        "/predict/batch",
        json={"records": [SAMPLE] * 101},
        headers=HEADERS,
    )
    assert too_large.status_code == 422


def test_model_info(client):
    response = client.get(
        "/model/info",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "test"


def test_service_loads_versioned_artifact(tmp_path):
    model = DummyClassifier(strategy="prior")
    model.fit(
        pd.DataFrame([SAMPLE, SAMPLE]),
        [0, 1],
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    joblib.dump(
        model,
        artifacts / "model_v3.joblib",
    )

    metadata = {
        "registry": {"version": "3"},
        "threshold": 0.45,
        "selected_model": "dummy_baseline",
        "test_metrics": {"recall": 0.5},
    }

    (artifacts / "model_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    service = ModelService.load(artifacts)
    output = service.predict([
        CustomerInput(**SAMPLE)
    ])

    assert len(output) == 1
    assert output[0].model_version == "3"
    assert output[0].churn_probability == 0.5

def test_monitoring_endpoints(client):
    assert client.get(
        "/monitoring/summary"
    ).status_code == 401

    summary = client.get(
        "/monitoring/summary",
        headers=HEADERS,
    )

    assert summary.status_code == 200
    assert "avg_latency_ms" in summary.json()

    drift = client.get(
        "/monitoring/drift",
        headers=HEADERS,
    )

    assert drift.status_code == 200
    assert "features" in drift.json()