
import json
from pathlib import Path

import pandas as pd


# Expected schema of the IBM Telco dataset.
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen",
    "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

# Initial acceptable ranges (business assumptions).
NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 100000),
}

# Accept only known categories during training.
CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {
        "Yes", "No", "No phone service"
    },
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {
        "Yes", "No", "No internet service"
    },
    "OnlineBackup": {
        "Yes", "No", "No internet service"
    },
    "DeviceProtection": {
        "Yes", "No", "No internet service"
    },
    "TechSupport": {
        "Yes", "No", "No internet service"
    },
    "StreamingTV": {
        "Yes", "No", "No internet service"
    },
    "StreamingMovies": {
        "Yes", "No", "No internet service"
    },
    "Contract": {
        "Month-to-month", "One year", "Two year"
    },
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_data(df: pd.DataFrame) -> dict:
    """Check data quality without changing the dataset."""

    errors = []
    warnings = []

    # Check dataset structure.
    missing = sorted(
        set(EXPECTED_COLUMNS) - set(df.columns)
    )

    if df.empty:
        errors.append("Dataset is empty")

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

    if errors:
        return {
            "status": "FAIL",
            "rows": len(df),
            "errors": errors,
            "warnings": warnings,
        }

    extra = sorted(
        set(df.columns) - set(EXPECTED_COLUMNS)
    )

    if extra:
        warnings.append(
            f"Unexpected columns (ignored in modeling): {extra}"
        )

    # Check missing and blank values.
    for column in EXPECTED_COLUMNS:
        raw = df[column].astype("string").str.strip()
        blank = raw.isna() | raw.eq("").fillna(False)
        count = int(blank.sum())

        if count:
            message = (
                f"{column}: {count} missing or blank values"
            )

            if column == "TotalCharges":
                warnings.append(message)
            else:
                errors.append(message)

    # Check duplicate customers.
    ids = (
        df["customerID"].astype("string").str.strip()
    )

    duplicates = int(ids.duplicated().sum())

    if duplicates:
        errors.append(
            f"Duplicate customer IDs: {duplicates}"
        )

    # Check numeric values and ranges.
    for column, (low, high) in NUMERIC_RANGES.items():
        raw = df[column].astype("string").str.strip()
        blank = raw.isna() | raw.eq("").fillna(False)

        values = pd.to_numeric(
            raw, errors="coerce"
        )

        invalid = int(
            (values.isna() & ~blank).sum()
        )

        outside = int(
            (
                values.notna()
                & ~values.between(low, high)
            ).sum()
        )

        if invalid:
            errors.append(
                f"{column}: {invalid} nonnumeric values"
            )

        if outside:
            errors.append(
                f"{column}: {outside} values outside "
                f"[{low}, {high}]"
            )

        if column in {"tenure", "SeniorCitizen"}:
            fractional = int(
                (
                    values.notna()
                    & (values % 1 != 0)
                ).sum()
            )

            if fractional:
                errors.append(
                    f"{column}: {fractional} noninteger values"
                )

    # Check categorical values.
    for column, allowed in CATEGORIES.items():
        values = (
            df[column]
            .dropna()
            .astype("string")
            .str.strip()
        )

        invalid_count = int(
            (
                ~values.isin(allowed)
                & values.ne("")
            ).sum()
        )

        if invalid_count:
            errors.append(
                f"{column}: {invalid_count} invalid categories"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(df),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    path = Path(
        "data/raw/Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(
        path, dtype={"customerID": "string"}
    )

    report = validate_data(df)

    output = Path(
        "artifacts/data_validation.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
