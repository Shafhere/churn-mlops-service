# Security and Engineering Review

Scope: an interview demonstration using the public IBM Telco sample dataset and a **single-instance deployment**. This document distinguishes implemented controls from known limits.

## Implemented

- Pydantic validation rejects missing and unexpected customer features and constrains numeric and categorical fields; batch endpoint caps at 100 records.
- API-key authentication uses a timing-safe comparison; `/health` is deliberately public, while prediction, model info and monitoring require a key.
- Streamlit has a separate demo password. Both secrets are generated locally and kept in `.env`, which Git and Docker ignore.
- Containers run as a non-root UID with `no-new-privileges`, and the model artifact directory is mounted read-only.
- Serving verifies the model artifact SHA-256 **before deserialization**, and checks the scikit-learn version recorded by training. This catches accidental changes; it is **not** signature-based provenance because an attacker able to edit both artifact and metadata could forge the hash.
- Docker API and UI ports bind to localhost; AWS optional proxy exposes public HTTP/HTTPS while the API itself remains private to the host/Docker network.
- Request logs and SQLite monitor store aggregate metadata, route names, HTTP statuses, prediction counts, latency and binned feature counts. They do **not** deliberately store individual customer request bodies or API keys.
- No raw customer dataset, secrets or local tracking databases should be committed to Git. MLflow is not exposed by the public serving stack.
- Prediction responses have `Cache-Control: no-store`; server input validation messages do not echo whole customer records.
- UI file uploads have a size and row limit; batch requests are split into smaller calls.

## Not implemented / trade-offs

- Shared passwords do not offer password lockout, per-user audit records, permissions or federated identity. Demo only; upgrade to identity-aware auth for real customer information.
- No global API rate limit, WAF, request-size enforcement at every network layer, secret rotation automation or security incident process. The Pydantic batch-size cap is not a complete denial-of-service defense.
- No threat-model assessment of all third-party dependencies or continuous vulnerability-scanning pipeline. Review CVEs and dependencies before public use.
- Monitoring SQLite is local; multiple replicas would require shared metrics / logging storage. It is not a multi-region/high-availability service.
- PSI compares **historical cumulative** serving distributions with training, with heuristics. It is not a proven detector of true performance degradation or a formally calibrated change-point test.
- Per-customer features are not a causal explanation. Replacing one input with a typical value may generate unrealistic combinations of customer features.
- Data quality checks are dataset-specific initial business assumptions. Real deployment requires a domain-approved schema, consent/retention policies and privacy impact assessment.
- AWS secrets in an `.env` file on the host are acceptable only for a limited demo with restricted host access; production should use AWS Secrets Manager/SSM with tightly scoped IAM.
- Model file remains pickle-based. Never expose a user-facing model upload endpoint or deserialize untrusted artifacts.

## Deployment checklist

1. Protect GitHub repo (private during review); check Git history for accidentally committed secrets; rotate any exposed credentials.
2. Run tests, lint and the real-model smoke test in **Python 3.12 with sklearn 1.7.2**.
3. Confirm no `.env`, customer data or `mlflow.db` enters the Docker build context or Git commit.
4. Generate new secrets on EC2; never reuse local development keys.
5. Use a public DNS record and the optional Caddy proxy for a valid HTTPS certificate. Only permit SSH from your IP, HTTP 80 and HTTPS 443 at AWS security group level; don't expose 8000 or 8501.
6. Verify `/health`, authenticated prediction, batch input, monitoring and 401/422 cases after deployment.
7. Set an AWS budget/billing alarm; understand running EC2, storage and public IPv4 billing; terminate resources when finished.
8. Before accepting live private customer data: add identity management, audit/retention policy, central monitoring, backups, rate limiting and authorized access controls.
