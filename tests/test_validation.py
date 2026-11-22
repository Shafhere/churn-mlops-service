
import pandas as pd

from modeling.validation import validate_data


def sample_data():
    # A single synthetic customer with valid values.
    return pd.DataFrame([{
        "customerID": "0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
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
        "MonthlyCharges": 35.5,
        "TotalCharges": "35.5",
        "Churn": "No",
    }])


def test_good_data_passes():
    result = validate_data(sample_data())
    assert result["status"] == "PASS"


def test_missing_column_fails():
    df = sample_data().drop(columns=["tenure"])
    assert validate_data(df)["status"] == "FAIL"


def test_duplicate_customer_fails():
    df = pd.concat(
        [sample_data(), sample_data()],
        ignore_index=True,
    )
    assert validate_data(df)["status"] == "FAIL"


def test_negative_charge_fails():
    df = sample_data()
    df.loc[0, "MonthlyCharges"] = -10
    assert validate_data(df)["status"] == "FAIL"


def test_invalid_churn_fails():
    df = sample_data()
    df.loc[0, "Churn"] = "Maybe"
    assert validate_data(df)["status"] == "FAIL"


def test_blank_total_charges_is_warning():
    df = sample_data()
    df.loc[0, "TotalCharges"] = " "

    result = validate_data(df)

    assert result["status"] == "PASS"
    assert any(
        "TotalCharges" in warning
        for warning in result["warnings"]
    )


def test_blank_contract_fails():
    df = sample_data()
    df.loc[0, "Contract"] = " "
    assert validate_data(df)["status"] == "FAIL"


def test_non_numeric_total_charges_fails():
    df = sample_data()
    df.loc[0, "TotalCharges"] = "not-a-number"
    assert validate_data(df)["status"] == "FAIL"
