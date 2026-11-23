# Interview Walkthrough — ChurnGuard

## 90-second explanation

“I built ChurnGuard as a small end-to-end churn prediction service, focusing on the engineering around the model rather than only accuracy. I started by validating the public Telco dataset, including its 11 blank TotalCharges values, and created a reproducible stratified split. I fitted imputers, scaling and one-hot encoding inside scikit-learn pipelines so the holdout data never influenced preprocessing. I compared a dummy baseline, logistic regression, class-weighted logistic regression and class-weighted random forest. I selected the candidate and decision threshold using the validation set and explicitly assumed false-negative and false-positive costs. The selected class-weighted logistic model is version 1 with threshold 0.37, tracked in MLflow, with a SHA-256 dataset fingerprint and model card.

FastAPI loads the exact versioned pipeline and rejects invalid inputs or missing API keys. Streamlit provides single and batch predictions, approximate local sensitivity factors and a monitoring dashboard. Aggregate SQLite records track latency, request errors, prediction counts and training-versus-serving PSI drift; they do not log customer rows. Tests cover quality rules and API contracts, and the service is packaged as two Docker containers with GitHub Actions lint and test checks. For an AWS demonstration I provide a single-EC2 Docker Compose deployment with HTTPS reverse proxy. It's a sound production-style demonstration, not a claim of enterprise autoscaling or measured live accuracy.”

## The actual flow

`CSV → validate → stratified train/validation/test → sklearn Pipeline → candidates → tune threshold on validation → test once → MLflow version + model card → API startup checks hash/version → API-key + Pydantic → probability + 0.37 decision → Streamlit → aggregate monitor + PSI`.

## What each file does

| File | One-sentence responsibility |
|---|---|
| `modeling/validation.py` | Data quality rules and report |
| `modeling/data.py` | Splits and leakage-safe preprocessing definition |
| `modeling/evaluation.py` | Precision/recall/F1/AUC, cost and threshold |
| `modeling/train.py` | Train, compare, select, log to MLflow, register and save the winner |
| `modeling/explain.py` | Global permutation importance |
| `modeling/model_card.py` | Generates model documentation from measured metrics |
| `modeling/verify.py` | Checks registry/local model consistency and source fingerprint |
| `api/schemas.py` | HTTP request and response contracts |
| `api/auth.py` | Shared API-key authentication |
| `api/service.py` | Load and verify model, run inference and approximate local sensitivity |
| `api/main.py` | HTTP routes and operational metrics middleware |
| `monitoring/reference.py` | Training-based binning and typical-value reference |
| `monitoring/store.py` | SQLite aggregation and PSI drift check |
| `ui/app.py` | User interface, CSV batch and monitoring display |

## Likely questions

**Why not pick by accuracy?** Majority-class accuracy can hide missed churners; I used cost on validation, plus PR summary, recall, ROC-AUC, F1 and calibration context.

**Why a 0.37 threshold?** On this validation set, that threshold minimized a hypothetical cost of 5 per missed churner and 1 per false alarm for the selected candidate. The actual business must validate those assumptions.

**Why three splits?** Train learns transformations and weights; validation selects model and threshold; untouched test assesses selected performance. Test results must not be reused to make new selection decisions.

**What if the customer schema changes?** Pydantic rejects unexpected/missing fields; the training validator detects schema drift. Approve a versioned schema/model update rather than silently adding fields.

**How do you keep reproducibility?** Pin Python and direct dependencies, fix random seeds, preserve dataset/training split hashes, model parameters, threshold, metrics and MLflow registry version. Cross-machine bit-identical artifacts are not guaranteed.

**Why MLflow?** Experiments record what was tried; registry versions identify the reviewed serving artifact. We don't require a live MLflow server to serve predictions.

**Why a separate API and UI?** The API owns validation, authentication and inference; the UI is a client. Model changes do not require rewriting UI logic if the API contract remains stable.

**How are missing TotalCharges handled?** Blank CSV strings become JSON null in the UI, Pydantic accepts null for that field, and the fitted training median imputer handles it. No upload-specific median is calculated.

**How do you know your model hasn't been replaced?** Artifact checksum verified on startup, metadata pins its model version and training hashes, and the sklearn runtime version is checked. Checksum is not signed provenance; trust the release source.

**What does PSI prove?** It signals distribution movement, not accuracy decay. Validated labels are required to assess whether the classifier still works.

**Can you scale to millions?** The stateless API can be replicated, but SQLite isn't shared and should move to centralized monitoring. A load balancer, autoscaling and object storage (e.g. AWS ECS + ALB + S3 + CloudWatch) would be the logical next infrastructure iteration.

**What if the API crashes?** Docker restarts the service; health checks detect a stopped/unhealthy API. Keep an approved previous image/model version for rollback. This is not high availability on a single EC2 instance.

**What if you had one extra day?** Run load/latency benchmarks, add identity-aware access and rate limits, monitor calibration/fairness with real labeled outcomes, and set formal retraining approval gates. I would not add automated retraining without monitored ground truth and rollback controls.

## Demo sequence (3–5 minutes)

1. Explain training comparison and actual holdout metrics in `docs/MODEL_CARD.md`.
2. Run `python -m pytest -q` and show a passing suite.
3. Run `python scripts/smoke_test.py` on the pinned Python 3.12 environment.
4. Show Swagger authorization, `POST /predict` and `GET /model/info` (version 1, threshold 0.37).
5. Open Streamlit, predict one public sample customer, upload the sample CSV and show monitoring counts + drift.
6. Explain `.github/workflows/ci.yml`, Docker Compose and the deployment trade-offs.
