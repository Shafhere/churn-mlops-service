# Verification Report — Final Release Build

## Checked in this environment (November 23, 2026)

- Inspected the uploaded release ZIP and kept the existing **v1** model, its real evaluation results and public dataset.
- Eliminated generated caches, duplicate unneeded model files, local monitoring databases, and the full local MLflow cache from the delivery ZIP.
- Compiled Python source without syntax errors.
- Ran **37 pytest tests successfully** using the container's available packages (Python 3.13, FastAPI 0.128.2, scikit-learn 1.8.0; the tests use synthetic fixtures).
- Confirmed the saved model artifact file's SHA-256 matches the metadata and the public training CSV matches its source fingerprint.
- Checked the Compose, AWS overlay and CI YAML files parse as YAML.
- Added a fail-fast runtime check when the scikit-learn version does not match the model's training version.

## Important limitations

- The saved model was trained with **Python 3.12.10 and scikit-learn 1.7.2**. The current test environment has **Python 3.13 and scikit-learn 1.8.0**, so a full saved-model prediction could **not** be completed here; it failed inside an incompatible sklearn `SimpleImputer`. This is why the release pins version 1.7.2 and refuses mismatched runtimes at startup.
- Docker Engine and AWS account credentials are not available in this test environment. Consequently, the two Docker images and public HTTPS EC2 deployment **have not been built or deployed here**. The provided configuration is prepared for the user's Docker/AWS verification.
- Ruff was not installed here, so final Ruff CLI validation must run in local development or GitHub Actions. Known reported lint issues from the prior source were corrected.
- The original user's Windows check reported successful `python -m modeling.verify`, model version 1, and threshold 0.37. That report is not independent verification of the new Docker release.
- The supplied MLflow database was intentionally excluded to keep deployment clean. The model metadata includes the registered model name/version/run ID, but the local registry UI cannot be reconstructed from the release ZIP alone without retraining or separately restoring the original MLflow DB/artifacts.

## Final checks before publicly sharing the service

1. With Python **3.12** installed, run `python -m pip install -r requirements.txt`, `python -m pytest -q`, `python -m ruff check api modeling monitoring ui tests scripts`, and `python scripts/smoke_test.py`.
2. Build Docker with `docker compose up -d --build`; confirm `docker compose ps` shows healthy API + UI.
3. Verify a single customer, a CSV batch, the monitoring dashboard and missing-key/malformed input errors.
4. Configure AWS EC2, HTTPS, DNS, credentials, and verify the real public URL. No public URL or live AWS deployment is claimed in this report.
