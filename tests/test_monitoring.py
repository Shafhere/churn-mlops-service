
from monitoring.store import (
    MonitorStore,
    population_stability_index,
)


def test_identical_distributions():
    score = population_stability_index(
        [40, 60], [40, 60]
    )
    assert abs(score) < 0.0001


def test_changed_distribution():
    score = population_stability_index(
        [80, 20], [20, 80]
    )
    assert score > 0.25


def test_monitoring_storage_and_drift(tmp_path):
    reference = {
        "features": {
            "Contract": {
                "type": "categorical",
                "counts": {
                    "Month-to-month": 80,
                    "Two year": 20,
                    "__MISSING__": 0,
                    "__OTHER__": 0,
                },
            },
        },
    }

    monitor = MonitorStore(
        tmp_path / "metrics.sqlite",
        reference,
    )

    # Not enough predictions to assess drift yet.
    assert (
        monitor.drift()["features"][0]["status"]
        == "insufficient_data"
    )

    # Store aggregate metrics for 50 predictions.
    monitor.record(
        route="/predict/batch",
        status=200,
        latency_ms=12.0,
        predictions=50,
        churn_yes=10,
        feature_counts={
            ("Contract", "Month-to-month"): 10,
            ("Contract", "Two year"): 40,
        },
    )

    # Record one failed request.
    monitor.record(
        route="/predict",
        status=422,
        latency_ms=5.0,
    )

    summary = monitor.summary()

    assert summary["requests"] == 2
    assert summary["errors"] == 1
    assert summary["predictions"] == 50
    assert summary["predicted_no_churn"] == 40

    drift = monitor.drift()["features"][0]

    assert drift["status"] == "alert"
