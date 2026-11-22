
import hashlib

import pandas as pd
from sklearn.dummy import DummyClassifier

from modeling.explain import get_feature_importance
from modeling.train import (
    file_sha256,
    training_sha256,
)


def test_file_hash_is_correct(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(b"sample-data")

    expected = hashlib.sha256(
        b"sample-data"
    ).hexdigest()

    assert file_sha256(path) == expected


def test_training_hash_changes_with_labels():
    X = pd.DataFrame({
        "tenure": [1, 5, 10],
        "MonthlyCharges": [20, 30, 40],
    })

    y1 = pd.Series([0, 0, 1])
    y2 = pd.Series([0, 1, 1])

    assert training_sha256(X, y1) != (
        training_sha256(X, y2)
    )


def test_feature_importance_has_original_columns():
    X = pd.DataFrame({
        "tenure": [1, 5] * 12,
        "MonthlyCharges": [20, 80] * 12,
    })

    y = [0, 1] * 12

    model = DummyClassifier(
        strategy="prior"
    ).fit(X, y)

    importance = get_feature_importance(
        model, X, y
    )

    names = {
        item["feature"]
        for item in importance
    }

    assert names == set(X.columns)
