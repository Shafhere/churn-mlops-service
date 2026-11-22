
import pandas as pd

from modeling.data import (
    make_preprocessor,
    split_data,
)
from modeling.validation import validate_data

# Synthetic customer, not real personal information.
ROW = {
    "customerID": "customer-0",
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
    "TotalCharges": "540",
    "Churn": "No",
}


def sample_data(n=40):
    records = []

    for i in range(n):
        row = ROW.copy()
        row["customerID"] = f"customer-{i}"
        row["Churn"] = (
            "Yes" if i % 3 == 0 else "No"
        )
        row["Contract"] = (
            "One year" if i % 2
            else "Month-to-month"
        )
        records.append(row)

    return pd.DataFrame(records)


def test_valid_data():
    assert validate_data(sample_data())["status"] == "PASS"


def test_missing_column():
    df = sample_data().drop(columns=["tenure"])
    assert validate_data(df)["status"] == "FAIL"


def test_duplicate_id():
    df = sample_data()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    assert validate_data(df)["status"] == "FAIL"


def test_negative_charge():
    df = sample_data()
    df.loc[0, "MonthlyCharges"] = -2
    assert validate_data(df)["status"] == "FAIL"


def test_blank_total_charge_warning():
    df = sample_data()
    df.loc[0, "TotalCharges"] = " "

    report = validate_data(df)

    assert report["status"] == "PASS"
    assert any(
        "TotalCharges" in warning
        for warning in report["warnings"]
    )


def test_invalid_target():
    df = sample_data()
    df.loc[0, "Churn"] = "Unknown"
    assert validate_data(df)["status"] == "FAIL"


def test_splits_are_disjoint_and_reproducible():
    first = split_data(sample_data())
    second = split_data(sample_data())

    assert first[0].index.equals(second[0].index)

    assert not (
        set(first[0].index) & set(first[1].index)
    )

    assert not (
        set(first[0].index) & set(first[2].index)
    )

    assert "customerID" not in first[0].columns
    assert "Churn" not in first[0].columns


def test_preprocessor_handles_missing_and_new_category():
    X_train, X_val, *_ = split_data(sample_data())

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train.iloc[
        0, X_train.columns.get_loc("TotalCharges")
    ] = float("nan")

    X_val.iloc[
        0, X_val.columns.get_loc("InternetService")
    ] = "Fiber optic"

    preprocessor = make_preprocessor()

    train = preprocessor.fit_transform(X_train)
    val = preprocessor.transform(X_val)

    assert train.shape[1] == val.shape[1]
