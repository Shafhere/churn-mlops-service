"""Smoke test a genuine saved model against the HTTP API, without AWS."""
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Do not depend on private deployment credentials for this local check.
    os.environ["CHURNGUARD_API_KEY"] = "temporary-smoke-test-key-only"
    with tempfile.TemporaryDirectory() as storage:
        os.environ["CHURNGUARD_MONITOR_DB_DIR"] = storage

        from api.main import app
        from api.schemas import CustomerInput

        sample = {
            "gender": "Female", "SeniorCitizen": 0,
            "Partner": "No", "Dependents": "No", "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.0, "TotalCharges": None,
        }
        customer = CustomerInput.model_validate(sample)
        assert customer.TotalCharges is None
        headers = {"X-API-Key": os.environ["CHURNGUARD_API_KEY"]}

        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200, health.text
            assert client.post("/predict", json=sample).status_code == 401
            info = client.get("/model/info", headers=headers)
            assert info.status_code == 200, info.text
            one = client.post("/predict", json=sample, headers=headers)
            assert one.status_code == 200, one.text
            assert 0 <= one.json()["churn_probability"] <= 1
            assert one.json()["threshold"] == info.json()["threshold"]
            batch = client.post(
                "/predict/batch", json={"records": [sample, sample]},
                headers=headers,
            )
            assert batch.status_code == 200, batch.text
            assert batch.json()["count"] == 2
            summary = client.get("/monitoring/summary", headers=headers)
            assert summary.status_code == 200, summary.text
            assert summary.json()["predictions"] == 3
        print("PASS: real model loaded, authenticated prediction and batch passed")
        print("Model version:", info.json()["model_version"])
        print("Threshold:", info.json()["threshold"])


if __name__ == "__main__":
    main()
