
"""Streamlit frontend for ChurnGuard."""

import hmac
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from api.schemas import CustomerInput

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env",
    override=False,
)

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📊",
    layout="wide",
)


def setting(name, default=""):
    """Read server-side environment or Streamlit secrets."""
    if os.getenv(name):
        return os.environ[name]

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


API_URL = setting(
    "CHURNGUARD_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_KEY = setting("CHURNGUARD_API_KEY")
UI_PASSWORD = setting("CHURNGUARD_UI_PASSWORD")


def api(method, endpoint, data=None):
    """Call the protected FastAPI service."""

    if not API_KEY:
        raise RuntimeError("API key is not configured")

    response = requests.request(
        method,
        API_URL + endpoint,
        headers={"X-API-Key": API_KEY},
        json=data,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"API returned HTTP {response.status_code}: "
            f"{response.text[:250]}"
        )

    return response.json()


# --------------------
# Demo UI authentication
# --------------------

if not UI_PASSWORD:
    st.error("Configure CHURNGUARD_UI_PASSWORD.")
    st.stop()

if not st.session_state.get("authenticated", False):
    st.title("ChurnGuard — Sign In")

    with st.form("login"):
        password = st.text_input(
            "Access password", type="password"
        )
        submitted = st.form_submit_button("Sign In")

    if submitted:
        if hmac.compare_digest(password, UI_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()


st.title("ChurnGuard MLOps")
st.caption("Customer Churn Prediction and Monitoring")

if st.sidebar.button("Sign Out"):
    st.session_state["authenticated"] = False
    st.rerun()


single_tab, batch_tab, monitor_tab = st.tabs([
    "Single Prediction",
    "Batch Prediction",
    "Monitoring",
])


# --------------------
# PAGE 1: Single prediction
# --------------------

with single_tab:
    st.subheader("Customer Churn Prediction")

    with st.form("customer_form"):
        left, right = st.columns(2)

        with left:
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox(
                "Dependents", ["No", "Yes"]
            )
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0, max_value=120, value=12
            )
            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )
            lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )

        with right:
            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )
            support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )
            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=45.0,
            )
            unknown_total = st.checkbox(
                "Total Charges Unavailable"
            )
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=540.0,
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": (
                None if unknown_total else total
            ),
        }

        try:
            st.session_state["single_result"] = api(
                "POST", "/predict", customer
            )
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]

        first, second = st.columns(2)

        first.metric(
            "Churn Probability",
            f"{result['churn_probability']:.1%}",
        )
        second.metric(
            "Prediction",
            result["prediction"],
        )

        st.caption(
            f"Model version: {result['model_version']} | "
            f"Decision threshold: {result['threshold']:.1%}"
        )

        factors = result.get("top_factors", [])

        if factors:
            st.subheader("Top Local Sensitivity Factors")
            st.dataframe(
                pd.DataFrame(factors),
                hide_index=True,
            )
            st.caption(
                "Approximate probability changes relative to "
                "training reference values. Not causal or "
                "additive SHAP explanations."
            )
        else:
            st.info(
                "Local explanations are not available."
            )


# --------------------
# PAGE 2: Batch prediction
# --------------------

with batch_tab:
    st.subheader("Batch CSV Prediction")

    st.write(
        "Upload up to 10,000 customer rows. "
        "Requests are processed in batches of 100."
    )

    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
    )

    if uploaded and st.button("Run Batch Prediction"):
        try:
            if uploaded.size > 5 * 1024 * 1024:
                raise ValueError("CSV exceeds the 5 MB limit")

            df = pd.read_csv(
                uploaded,
                dtype={"customerID": "string"},
            )

            # Convert blank TotalCharges to missing values.
            # The trained ML pipeline will impute them.

            charges = df["TotalCharges"].astype("string").str.strip()

            # Empty strings become missing values.
            charges = charges.replace("", pd.NA)

            # Convert valid numeric strings to numbers.
            # Reject unexpected text rather than silently cleaning it.
            df["TotalCharges"] = pd.to_numeric(
                charges,
                errors="raise",
            )


            fields = list(CustomerInput.model_fields)

            missing = set(fields) - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing columns: {sorted(missing)}"
                )

            if not 1 <= len(df) <= 10000:
                raise ValueError(
                    "CSV must contain 1 to 10,000 rows"
                )

            extra = (
                set(df.columns)
                - set(fields)
                - {"customerID", "Churn"}
            )

            if extra:
                raise ValueError(
                    f"Unexpected columns: {sorted(extra)}"
                )

            # Missing values become JSON null.
            records = json.loads(
                df[fields].to_json(orient="records")
            )

            predictions = []

            for start in range(0, len(records), 100):
                chunk = records[start:start + 100]

                result = api(
                    "POST",
                    "/predict/batch",
                    {"records": chunk},
                )

                predictions.extend(
                    result["predictions"]
                )

            output = df.copy()

            output["churn_probability"] = [
                p["churn_probability"]
                for p in predictions
            ]
            output["predicted_churn"] = [
                p["prediction"]
                for p in predictions
            ]

            st.session_state["batch_csv"] = (
                output.to_csv(index=False)
            )

            st.success(
                f"Processed {len(predictions)} customers"
            )

            st.dataframe(
                output.drop(
                    columns=["customerID"],
                    errors="ignore",
                ).head(30)
            )

        except (
            ValueError,
            requests.RequestException,
            RuntimeError,
        ) as error:
            st.error(str(error))

    if st.session_state.get("batch_csv"):
        st.download_button(
            "Download Prediction Results",
            st.session_state["batch_csv"],
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    st.caption(
        "Uploaded data is processed in the app session. "
        "Do not use private customer data on an "
        "unapproved public deployment."
    )


# --------------------
# PAGE 3: Monitoring
# --------------------

with monitor_tab:
    st.subheader("Model Monitoring")

    if st.button("Refresh Monitoring"):
        try:
            st.session_state["monitor_summary"] = api(
                "GET", "/monitoring/summary"
            )
            st.session_state["monitor_drift"] = api(
                "GET", "/monitoring/drift"
            )
        except (
            requests.RequestException,
            RuntimeError,
        ) as error:
            st.error(str(error))

    summary = st.session_state.get("monitor_summary")

    if summary:
        a, b, c, d = st.columns(4)

        a.metric("Requests", summary["requests"])
        b.metric(
            "Error Rate",
            f"{summary['error_rate']:.1%}",
        )
        c.metric(
            "Average Latency",
            f"{summary['avg_latency_ms']:.1f} ms",
        )
        d.metric(
            "Predictions",
            summary["predictions"],
        )

        distribution = pd.DataFrame({
            "Prediction": ["Churn", "No Churn"],
            "Count": [
                summary["predicted_churn"],
                summary["predicted_no_churn"],
            ],
        })

        st.subheader("Prediction Distribution")
        st.bar_chart(
            distribution.set_index("Prediction")
        )

    drift = st.session_state.get("monitor_drift")

    if drift:
        st.subheader("Feature Drift (PSI)")
        st.dataframe(
            pd.DataFrame(drift["features"]),
            hide_index=True,
        )
        st.caption(
            "At least 50 observations are required. "
            "PSI indicates distribution change, "
            "not confirmed model-performance decline."
        )
