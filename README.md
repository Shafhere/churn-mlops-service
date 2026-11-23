# ChurnGuard MLOps

**A production-style telecom churn prediction service and monitoring demonstration.**  
Python 3.12 · scikit-learn · MLflow · FastAPI · Streamlit · SQLite · Docker · GitHub Actions

> Status: **deployment candidate**, not a validated enterprise production system. The supplied model is the original trained **version 1** and has not been retrained during cleanup.

## What it does

An authorized user enters customer service/billing information or uploads a CSV. FastAPI validates the inputs, runs the saved preprocessing-and-classifier pipeline, compares the probability with a **0.37** decision threshold, and returns a prediction. Streamlit displays results and monitoring. The application stores only aggregate operational records for drift/latency; it does not log customer input rows.

```text
Public browser (HTTPS on AWS) → Caddy → Streamlit UI
                                       ↓ (private Docker network)
                                   FastAPI API-key endpoint
                                       ↓
                        Validated features → versioned sklearn Pipeline
                                       ↓
                        Probability → 0.37 decision threshold
                                       ↓
                             Aggregate SQLite monitoring
```

**Single-instance design:** components are separated for maintainability, not separate distributed microservices. Horizontal scaling would require shared monitoring storage and a deployment orchestrator.

## Repository layout

| Path | What it is |
|---|---|
| `modeling/` | Validate, preprocess, split data, train/compare models, evaluate, MLflow, version and explain |
| `api/` | Pydantic schemas, API-key authentication, versioned artifact loader, inference and routes |
| `ui/app.py` | Streamlit form, batch CSV upload/download, monitoring dashboard |
| `monitoring/` | Training reference, aggregate SQLite metrics, PSI feature-drift check |
| `tests/` | Unit/API tests using synthetic inputs; does not require the real raw dataset |
| `artifacts/` | Demo release model v1, model metadata and monitoring baseline (included in release ZIP; ignored by Git) |
| `data/raw/` | Public Telco sample CSV in release ZIP (ignored by Git) |
| `docs/` | Model card, deployment, security and interview explanation |
| `deploy/` | Optional Caddy HTTPS reverse proxy for AWS EC2 |
| `.github/workflows/ci.yml` | Lint, tests and syntax checks on push/PR |
| `scripts/` | Generate credentials and run a real-model smoke test |
| `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml` | Containerized local serving |

## Model and data

Dataset: public [IBM Telco Customer Churn sample](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7,043 rows and 21 original columns. The original CSV has **11 blank `TotalCharges` values**. Customer IDs and the churn label are not model features.

- Data quality: schema, mandatory fields, duplicates, categorical values, numeric ranges.
- Leakage prevention: stratified train/validation/test splits; preprocessing fitted only on training data in each classifier pipeline.
- Candidates: dummy prior baseline, logistic regression, class-weighted logistic regression, class-weighted random forest.
- Imbalance: stratified splits and a direct comparison of weighted/unweighted models.
- Evaluation: precision, recall, F1, average precision (PR summary), ROC-AUC, Brier score, and hypothetical business cost.
- Threshold: chosen using validation data under assumed false-negative cost = 5 and false-positive cost = 1.
- MLflow: tracked candidate experiments and registered version 1. Tracked results, source SHA-256 and training-split SHA-256 are preserved in metadata.
- Explanation: global permutation importance; per-customer display uses **one-feature-at-a-time sensitivity** (neither SHAP nor causal explanation).

### Actual supplied holdout test results (model v1)

| Metric | Value |
|---|---:|
| Selected model | Class-weighted Logistic Regression |
| Threshold | **0.37** |
| Average Precision (PR summary) | **0.6586** |
| ROC-AUC | **0.8446** |
| Precision | **0.4540** |
| Recall | **0.8790** |
| F1 | **0.5988** |
| Accuracy | **0.6868** |
| Brier score | **0.1630** |
| FP / FN | **297 / 34** |
| Estimated cost | **467 hypothetical cost units** |

These values were read from the supplied `model_metadata.json` and are **not measurements from live production traffic**. See [MODEL_CARD.md](docs/MODEL_CARD.md).

## Requirements

- Python **3.12** for direct local serving (the included model was trained with Python 3.12.10 and **scikit-learn 1.7.2**).
- Docker Engine with Compose plugin for container execution.
- No AWS credentials are needed for local testing.
- Exact top-level dependency versions are pinned in the respective requirements files. This is *not* an immutable full transitive lock of all Linux wheels.

**Important:** The serialized model intentionally refuses to load under a different scikit-learn version. `joblib`/pickle artifacts must come from trusted sources and are not safe to load from arbitrary user uploads.

## Run locally — Windows / macOS / Linux

```bash
python -m venv .venv
# Windows CMD: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/create_env.py
python -m pytest -q
python -m ruff check api modeling monitoring ui tests scripts
python scripts/smoke_test.py
```

The credential generator creates `.env` with two *different* random secrets and never overwrites an existing one. Open it locally to see the UI password; do not commit or share it.

Start separate terminals:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run ui/app.py --server.port 8501
```

- UI: <http://localhost:8501>
- Swagger: <http://127.0.0.1:8000/docs> (authorize using your private API key)
- Health: <http://127.0.0.1:8000/health>

## Docker Compose — recommended

```bash
python scripts/create_env.py
# Run from the extracted project root with Docker running:
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

The UI is at <http://localhost:8501>. Both API and UI ports are bound to `127.0.0.1` on the host. The model artifacts are mounted read-only; the monitoring database is stored in a persistent named volume. To stop without erasing monitoring: `docker compose down`.

The Docker build does **not** retrain the model or need MLflow. It installs only serving-time dependencies. It does not bundle `.env` or customer data into images.

## API contracts

| Method | Endpoint | Auth |
|---|---|---|
| GET | `/health` | Public |
| GET | `/model/info` | API key |
| POST | `/predict` | API key |
| POST | `/predict/batch` | API key (1–100 records/request) |
| GET | `/monitoring/summary` | API key |
| GET | `/monitoring/drift` | API key |

The frontend chunks up to 10,000 uploaded rows into API requests of 100. For blank `TotalCharges`, the UI sends JSON `null`; the fitted pipeline imputes using training statistics. Invalid fields are rejected with HTTP 422. Protected endpoints use `X-API-Key`. The health endpoint is intentionally public for health probes.

## Testing and CI

```bash
python -m pytest -q
python -m ruff check api modeling monitoring ui tests scripts
python scripts/smoke_test.py
```

GitHub Actions runs lint and synthetic-data unit/API tests on each main push and pull request. It does not upload raw data, deploy cloud infrastructure, or retrain the model automatically.

## Model training / MLflow (optional reproduction)

The release includes the public sample CSV and model v1. Training is **not required for hosting**. To train a new candidate locally:

```bash
python -m modeling.validation
python -m modeling.train
python -m monitoring.reference
python -m modeling.verify
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Rerunning training may register a **new version** and generate new metadata. Review and approve a replacement model before redeploying it; do not assume a larger version number is automatically better.

## AWS deployment

See [DEPLOY_AWS.md](docs/DEPLOY_AWS.md). The release includes an optional Caddy HTTPS proxy and a single-instance EC2 Compose deployment. A public, certificate-valid HTTPS URL requires an AWS instance, public DNS name, pointing DNS and opening only required ports. Those resources and external credentials are not included in this ZIP.

## Security and limitations

See [SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md) for implemented controls, gaps and future improvements.

- This is a working **engineering demonstration candidate**, not a claim of enterprise availability or GDPR/PCI compliance.
- API key and UI password are shared credentials, **not** user-specific RBAC, SSO or abuse-resistant identity management.
- No automatic retraining or A/B experiment is included. These are optional assignment bonuses; adding them without quality gates would be counterproductive.
- Monitoring is single-instance and local, tracking operational signals and PSI rather than confirmed model performance against new labeled outcomes.
- No true scale/load benchmarks or AWS public deployment have been verified in this release environment.
- Model may not generalize to different telecom providers, time periods or populations without validation.
