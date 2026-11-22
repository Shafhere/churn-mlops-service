
import numpy as np

from sklearn.datasets import make_classification

from modeling.evaluation import (
    evaluate,
    select_threshold,
    threshold_metrics,
)
from modeling.train import get_candidates


def test_business_cost():
    # One false positive and one false negative.
    actual = [0, 0, 1, 1]
    probabilities = [0.2, 0.7, 0.3, 0.8]

    result = threshold_metrics(
        actual, probabilities, 0.5
    )

    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["business_cost"] == 6


def test_threshold_selection():
    actual = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.3, 0.4]

    threshold = select_threshold(
        actual, probabilities
    )

    result = evaluate(
        actual, probabilities, threshold
    )

    assert result["business_cost"] == 0
    assert 0.2 < threshold <= 0.3


def test_model_comparison_contains_baseline():
    models = get_candidates()

    assert "dummy_baseline" in models
    assert "logistic_regression" in models
    assert "logistic_balanced" in models
    assert "random_forest_balanced" in models


def test_all_candidates_produce_probabilities():
    # Synthetic data avoids using customer information.
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=4,
        weights=[0.75, 0.25],
        random_state=42,
    )

    for model in get_candidates().values():
        model.fit(X[:100], y[:100])

        probabilities = model.predict_proba(
            X[100:]
        )[:, 1]

        assert len(probabilities) == 50
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
