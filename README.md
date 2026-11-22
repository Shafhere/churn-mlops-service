
# ChurnGuard MLOps

An end-to-end machine learning prediction service for
telecom customer churn, focusing on data quality,
reproducibility, model management and secure serving.

## Project Objective

Predict the probability of customer churn using a
production-style machine learning pipeline.

The project demonstrates engineering discipline beyond
model accuracy, including validation, experiment tracking,
versioning, API development, testing and monitoring.

## Technology Stack

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI and Pydantic
- Pytest
- Git

Planned: Streamlit, Docker, GitHub Actions and AWS.

## Project Structure

modeling/   - Validation, preprocessing, training, evaluation
api/        - Authentication, schemas, prediction service
ui/         - Streamlit interface (planned)
monitoring/ - Logging and drift detection (planned)
tests/      - Automated testing
data/       - Local dataset
artifacts/  - Saved models and metadata
docs/       - Model card and technical documentation

## Machine Learning Workflow

1. Validate the original dataset.
2. Create stratified train/validation/test splits.
3. Build leakage-safe preprocessing pipelines.
4. Train a dummy baseline and candidate classifiers.
5. Compare models using relevant classification metrics.
6. Select a threshold using assumed business costs.
7. Evaluate the selected model on held-out test data.
8. Track experiments using MLflow.
9. Register the model and generate versioned metadata.
10. Generate a model card and feature importance report.

## Dataset

IBM Telco Customer Churn

Target: Churn (Yes/No)

Customer IDs are excluded from model features.

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Reproducibility

- Fixed random seed: 42
- Versioned dependencies
- SHA-256 source dataset fingerprint
- Training split fingerprint
- Stored model parameters and evaluation metrics
- Registered MLflow model versions

## FastAPI Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | API health |
| GET | /model/info | Model version and metrics |
| POST | /predict | Single prediction |
| POST | /predict/batch | Batch predictions |

All endpoints except /health require an API key.

## Security

- API-key authentication for protected endpoints
- Pydantic request validation
- Secrets stored outside source code
- Customer identifiers excluded from model inputs
- Local datasets excluded from Git
- Restricted batch request size
- No customer values echoed in validation errors
- Trusted model artifacts only

HTTPS, cloud access controls and monitoring
will be completed before public deployment.

## Local Setup

Create and activate the environment:

    py -3.12 -m venv .venv
    .venv\Scripts\activate

Install dependencies:

    python -m pip install -r requirements.txt

Create .env and set CHURNGUARD_API_KEY.

Train the model:

    python -m modeling.train

Verify the registered model:

    python -m modeling.verify

Run the API:

    python -m uvicorn api.main:app --reload --port 8000

API documentation:

    http://127.0.0.1:8000/docs

## Automated Tests

Run all tests:

    python -m pytest -v

## Model Documentation

See docs/MODEL_CARD.md for:

- Model purpose
- Dataset information
- Candidate comparison
- Selected threshold
- Evaluation results
- Feature importance
- Model limitations
- Version and training data fingerprint

## Current Limitations

- Public sample dataset, not production customer data
- Illustrative business-cost assumptions
- No real customer outcome monitoring yet
- No user-specific authorization
- Streamlit UI and drift monitoring pending
- Cloud deployment pending

## Next Development Stages

1. Streamlit UI with single and batch prediction
2. Request logging, metrics and data drift monitoring
3. Docker and Docker Compose
4. GitHub Actions CI
5. AWS deployment and security checks
6. End-to-end testing and documentation


## Streamlit and Monitoring

The Streamlit application provides:

- Single-customer churn prediction
- Churn probability and model threshold
- Approximate local feature sensitivity
- Batch CSV upload and downloadable predictions
- Aggregate request and error metrics
- Average prediction-service latency
- Prediction distribution
- PSI-based feature drift warnings

### Running the Application

Start FastAPI:

    python -m uvicorn api.main:app --reload --port 8000

Start Streamlit in another terminal:

    python -m streamlit run ui/app.py --server.port 8501

Open:

    http://localhost:8501

### Monitoring Design

Training reference distributions are generated using:

    python -m monitoring.reference

Operational metrics are stored in a local SQLite database,
separated by model version.

Monitoring stores aggregate feature counts, not original
customer records.

The drift check requires at least 50 predictions.
Its PSI thresholds are operational heuristics.

### Current Limitations

- Monitoring uses local SQLite; multi-instance deployments
  require centralized metric storage.
- This demonstration uses shared-secret UI authentication.
- No production identity provider is connected.
- Model quality cannot be measured continuously without
  new labeled customer outcomes.
- Local sensitivity explanations are approximate and
  should not be interpreted as causal effects.
- AWS deployment and infrastructure security are pending.
``` id="r4k8mz"

Then:

```cmd
python -m pip freeze > requirements.lock.txt
git add api ui monitoring tests README.md requirements.txt requirements.lock.txt .env.example
git status
```

Review the staged files and make sure no secrets, customer CSVs or SQLite databases are included.

```cmd
git commit -m "feat: add Streamlit interface and model monitoring"
```

---

## Part 2 — Interview preparation

| Likely question | What you should explain |
|---|---|
| Why Streamlit? | It provides a quick, understandable interface for prediction and monitoring while keeping ML inference in FastAPI |
| Why not load the model directly in Streamlit? | Centralized inference avoids duplicating model-loading logic, authentication rules and version selection |
| How is latency tracked? | Middleware measures request duration and stores aggregate timing metrics |
| How are errors tracked? | Monitoring counts HTTP error responses and calculates an error rate |
| What is data drift? | A change in the distribution of incoming features compared with training data |
| Does PSI prove the model is inaccurate? | No. It indicates potential distribution change; labeled outcomes are needed to evaluate actual predictive performance |
| How can monitoring scale? | Move metrics to centralized storage and use tools such as CloudWatch or Prometheus while scaling stateless API instances |
| How are customer details protected? | Validate inputs, authenticate endpoints, avoid raw-data logging, restrict access and use HTTPS for deployment |

## What comes next?

<box border radius="lg" padding={3} gap={2}>
  <title size="lg">Part 3 — Deployment and Final Engineering</title>
  <text color="secondary" size="sm">After Part 2 passes locally</text>
  <list marker="number" gap={1}>
    <list-item>Dockerfile, `.dockerignore` and Docker Compose</list-item>
    <list-item>GitHub repository, Actions CI, linting and automated tests</list-item>
    <list-item>Container resource limits and performance testing</list-item>
    <list-item>AWS infrastructure, HTTPS, secrets and logging</list-item>
    <list-item>Public deployment and live endpoint verification</list-item>
    <list-item>Security review, complete README and final interview walkthrough</list-item>
  </list>
</box>

**Your next checkpoint:** Run both servers, complete one single prediction, one batch prediction, and refresh the monitoring page. Once these work, Part 2's main functionality is complete and we can move directly into Docker, GitHub Actions and AWS deployment.