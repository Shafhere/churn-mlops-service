
import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
REPORT_PATH = Path("artifacts/data_validation.json")

# Columns required by the training dataset.
EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def validate_data(df: pd.DataFrame) -> dict:
    """Validate raw data without modifying customer records."""

    errors = []
    warnings = []

    # 1. Check required columns
    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        errors.append("Dataset has no rows")

    # Stop detailed checks when required columns are absent.
    if not missing_columns and not df.empty:

        extra_columns = set(df.columns) - EXPECTED_COLUMNS

        if extra_columns:
            warnings.append(
                f"Extra columns: {sorted(extra_columns)}"
            )

        # 2. Check missing and blank required values
        # TotalCharges is handled separately.
        for column in EXPECTED_COLUMNS - {"TotalCharges"}:
            text = df[column].astype("string").str.strip()

            missing_count = int(
                (text.isna() | text.eq("")).sum()
            )

            if missing_count:
                errors.append(
                    f"{column}: {missing_count} missing/blank values"
                )

        # 3. Check duplicate customer identifiers
        customer_ids = (
            df["customerID"].astype("string").str.strip()
        )

        duplicates = int(
            customer_ids.dropna().duplicated().sum()
        )

        if duplicates:
            errors.append(
                f"Duplicate customer IDs: {duplicates}"
            )

        # 4. Validate numeric fields and reasonable ranges
        # These upper limits are configurable business assumptions.
        numeric_rules = [
            ("tenure", 0, 120),
            ("MonthlyCharges", 0, 1000),
            ("TotalCharges", 0, 100000),
            ("SeniorCitizen", 0, 1),
        ]

        for column, lower, upper in numeric_rules:
            raw_values = (
                df[column].astype("string").str.strip()
            )

            numbers = pd.to_numeric(
                raw_values, errors="coerce"
            )

            blanks = (
                raw_values.isna() | raw_values.eq("")
            )

            invalid = int(
                (numbers.isna() & ~blanks).sum()
            )

            outside = int(
                (
                    numbers.notna()
                    & ~numbers.between(lower, upper)
                ).sum()
            )

            if invalid:
                errors.append(
                    f"{column}: {invalid} non-numeric values"
                )

            if outside:
                errors.append(
                    f"{column}: {outside} values outside "
                    f"{lower}-{upper}"
                )

            # These columns must contain whole numbers.
            if column in {"tenure", "SeniorCitizen"}:
                fractions = int(
                    (
                        numbers.notna()
                        & (numbers % 1 != 0)
                    ).sum()
                )

                if fractions:
                    errors.append(
                        f"{column}: {fractions} non-integer values"
                    )

            # Blank TotalCharges may be handled in preprocessing.
            if column == "TotalCharges":
                blank_count = int(blanks.sum())

                if blank_count:
                    warnings.append(
                        f"TotalCharges: {blank_count} blank "
                        "values; review during preprocessing"
                    )

        # 5. Check important categorical values
        category_rules = {
            "Churn": {"Yes", "No"},
            "Contract": {
                "Month-to-month", "One year", "Two year"
            },
        }

        for column, allowed in category_rules.items():
            observed = set(
                df[column]
                .dropna()
                .astype("string")
                .str.strip()
            )

            unexpected = sorted(observed - allowed)

            if unexpected:
                errors.append(
                    f"{column}: {len(unexpected)} "
                    "unexpected category value(s)"
                )

    # 6. Create a summary with no customer-level data
    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(df),
        "columns": len(df.columns),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    report = validate_data(df)

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Status: {report['status']}")
    print(
        f"Rows: {report['rows']} | "
        f"Columns: {report['columns']}"
    )

    for message in report["errors"]:
        print("ERROR:", message)

    for message in report["warnings"]:
        print("WARNING:", message)

    print(f"Report: {REPORT_PATH}")

    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
