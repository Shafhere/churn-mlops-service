
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from modeling.validation import (
    EXPECTED_COLUMNS,
    validate_data,
)


RANDOM_STATE = 42

RAW_DATA = Path(
    "data/raw/Telco-Customer-Churn.csv"
)

NUMERIC = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Exclude the ID and target from features.
FEATURES = [
    column
    for column in EXPECTED_COLUMNS
    if column not in {"customerID", "Churn"}
]

CATEGORICAL = [
    column
    for column in FEATURES
    if column not in NUMERIC
]


def split_data(df: pd.DataFrame):
    """Validate and create reproducible data splits."""

    report = validate_data(df)

    if report["status"] != "PASS":
        raise ValueError(
            f"Data validation failed: {report['errors']}"
        )

    # Work on a copy. Never modify the raw input.
    clean = df.copy()

    clean["TotalCharges"] = pd.to_numeric(
        clean["TotalCharges"],
        errors="coerce",
    )

    X = clean[FEATURES]

    # No = 0, Yes = 1
    y = clean["Churn"].map({
        "No": 0,
        "Yes": 1,
    }).astype(int)

    # First split: 70% training, 30% temporary.
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Split temporary data into validation and test.
    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    )


def make_preprocessor():
    """Build reusable preprocessing transformations."""

    # Numeric preprocessing
    numeric_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    # Categorical preprocessing
    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    # Apply the correct steps to each column type.
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def main():
    df = pd.read_csv(
        RAW_DATA,
        dtype={"customerID": "string"},
    )

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    ) = split_data(df)

    # Fit only on training data.
    # Step 4 will include this in the model Pipeline.
    preprocessor = make_preprocessor()

    train_matrix = preprocessor.fit_transform(
        X_train
    )

    val_matrix = preprocessor.transform(
        X_val
    )

    test_matrix = preprocessor.transform(
        X_test
    )

    print(
        f"Samples: train={len(y_train)}, "
        f"validation={len(y_val)}, "
        f"test={len(y_test)}"
    )

    print(
        f"Churn rate: train={y_train.mean():.3f}, "
        f"validation={y_val.mean():.3f}, "
        f"test={y_test.mean():.3f}"
    )

    print(
        "Encoded feature count:",
        train_matrix.shape[1],
    )

    print(
        "Encoded shapes:",
        train_matrix.shape,
        val_matrix.shape,
        test_matrix.shape,
    )

    print(
        "Customer ID and target excluded from features."
    )


if __name__ == "__main__":
    main()
