"""Operational monitoring with aggregate-only storage."""

import logging
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from monitoring.reference import bucket

logger = logging.getLogger(__name__)


def population_stability_index(expected_counts, actual_counts):
    """Compare training and serving distributions."""

    expected = np.asarray(expected_counts, dtype=float)
    actual = np.asarray(actual_counts, dtype=float)

    if expected.sum() <= 0 or actual.sum() <= 0:
        return None

    # Smooth empty bins to avoid division by zero.
    expected = np.clip(expected / expected.sum(), 1e-4, None)
    actual = np.clip(actual / actual.sum(), 1e-4, None)

    expected /= expected.sum()
    actual /= actual.sum()

    return float(np.sum(
        (actual - expected) * np.log(actual / expected)
    ))


class MonitorStore:
    def __init__(self, path, reference=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reference = reference or {"features": {}}

        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    route TEXT,
                    status INTEGER,
                    latency_ms REAL,
                    predictions INTEGER,
                    churn_yes INTEGER
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS hist (
                    feature TEXT,
                    bucket TEXT,
                    count INTEGER,
                    PRIMARY KEY(feature, bucket)
                )
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(
            self.path, timeout=5
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def feature_counts(self, records):
        """Count input feature bins without saving records."""

        counts = Counter()

        for record in records:
            row = (
                record.model_dump()
                if hasattr(record, "model_dump")
                else record
            )

            for name, spec in self.reference.get(
                "features", {}
            ).items():
                label = bucket(row.get(name), spec)
                counts[(name, label)] += 1

        return counts

    def record(
        self, route, status, latency_ms,
        predictions=0, churn_yes=0, feature_counts=None
    ):
        """Record one request and its aggregate predictions."""

        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO requests
                    (route, status, latency_ms, predictions, churn_yes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        route, status, latency_ms,
                        predictions, churn_yes
                    ),
                )

                for (feature, label), count in (
                    feature_counts or {}
                ).items():
                    connection.execute(
                        """
                        INSERT INTO hist(feature, bucket, count)
                        VALUES (?, ?, ?)
                        ON CONFLICT(feature, bucket)
                        DO UPDATE SET count=count+excluded.count
                        """,
                        (feature, label, count),
                    )

        except sqlite3.Error:
            logger.exception("Monitoring write failed")

    def summary(self):
        """Return operational metrics."""

        with self.connect() as connection:
            count, errors, latency, predictions, yes = (
                connection.execute("""
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(
                            CASE WHEN status >= 400 THEN 1 ELSE 0 END
                        ), 0),
                        COALESCE(AVG(latency_ms), 0),
                        COALESCE(SUM(predictions), 0),
                        COALESCE(SUM(churn_yes), 0)
                    FROM requests
                """).fetchone()
            )

        return {
            "requests": count,
            "errors": errors,
            "error_rate": round(
                errors / count, 4
            ) if count else 0.0,
            "avg_latency_ms": round(latency, 1),
            "predictions": predictions,
            "predicted_churn": yes,
            "predicted_no_churn": predictions - yes,
        }

    def drift(self, minimum_samples=50):
        """Compute per-feature PSI against training data."""

        with self.connect() as connection:
            actual = {
                (feature, label): count
                for feature, label, count in connection.execute(
                    "SELECT feature, bucket, count FROM hist"
                ).fetchall()
            }

        results = []

        for name, spec in self.reference.get(
            "features", {}
        ).items():
            labels = list(spec["counts"])

            expected_counts = [
                spec["counts"][label] for label in labels
            ]
            actual_counts = [
                actual.get((name, label), 0)
                for label in labels
            ]

            samples = sum(actual_counts)

            if samples < minimum_samples:
                score = None
                status = "insufficient_data"
            else:
                score = population_stability_index(
                    expected_counts, actual_counts
                )

                if score < 0.1:
                    status = "stable"
                elif score < 0.25:
                    status = "watch"
                else:
                    status = "alert"

            results.append({
                "feature": name,
                "samples": samples,
                "psi": (
                    round(score, 4)
                    if score is not None else None
                ),
                "status": status,
            })

        return {
            "minimum_samples": minimum_samples,
            "features": results,
        }