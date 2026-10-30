
import json

import pandas as pd
import pytest

import finguard.orchestration.pipeline as pipeline


def test_spark_failure_records_failed_run_and_preserves_current_batch(
    tmp_path, monkeypatch
):
    paths = {
        "input_csv": str(tmp_path / "input.csv"),
        "quarantine_dir": str(tmp_path / "quarantine"),
        "processed_dir": str(tmp_path / "processed"),
        "reports_dir": str(tmp_path / "reports"),
        "monitoring_dir": str(tmp_path / "monitoring"),
        "sqlite_path": str(tmp_path / "runs.db"),
        "scored_dir": str(tmp_path / "scored"),
        "model_dir": str(tmp_path / "models"),
    }

    settings = {
        "paths": paths,
        "project": {
            "batch_id": "test-batch",
            "random_seed": 42,
        },
        "validation": {
            "min_amount": 0.01,
            "max_amount": 100000,
            "supported_currencies": ["INR"],
            "allowed_categories": ["groceries"],
            "allowed_payment_methods": ["card"],
            "max_rejection_rate": 0.10,
        },
        "spark": {
            "app_name": "FinGuard-Test",
            "shuffle_partitions": 2,
        },
        "model": {
            "version": "test-model",
            "high_risk_threshold": 0.8,
        },
    }

    input_data = pd.DataFrame(
        [
            {
                "transaction_id": "TX001",
                "transaction_timestamp": "2026-10-30T10:00:00Z",
                "amount": 500.0,
                "currency": "INR",
                "merchant_category": "groceries",
                "payment_method": "card",
                "customer_id": "C001",
                "distance_from_home_km": 2.5,
                "transactions_last_24h": 3,
                "is_fraud": 0,
            }
        ]
    )

    # Set up a previously published batch pointer.
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir(parents=True)

    pointer_path = scored_dir / "current.json"

    previous_pointer = {
        "batch_id": "previous-batch",
        "run_id": "previous-run",
        "status": "published",
    }

    pointer_path.write_text(
        json.dumps(previous_pointer),
        encoding="utf-8",
    )

    original_pointer = pointer_path.read_text(encoding="utf-8")

    recorded_runs = []

    monkeypatch.setattr(pipeline, "config", lambda config_path: settings)
    monkeypatch.setattr(pipeline, "logging_setup", lambda: None)
    monkeypatch.setattr(pipeline, "read_csv", lambda path: input_data.copy())

    monkeypatch.setattr(
        pipeline,
        "run_record",
        lambda db_path, record: recorded_runs.append(record),
    )

    def fail_spark(*args, **kwargs):
        raise RuntimeError("Simulated Spark processing failure")

    monkeypatch.setattr(pipeline, "run_spark", fail_spark)

    with pytest.raises(
        RuntimeError,
        match="Simulated Spark processing failure",
    ):
        pipeline.run_pipeline(config_path="unused-test-config.yaml")

    # Verify failed execution was recorded.
    assert len(recorded_runs) == 1

    failed_run = recorded_runs[0]

    assert failed_run["status"] == "failed"
    assert failed_run["error"] == "Simulated Spark processing failure"
    assert failed_run["input_count"] == 1
    assert failed_run["accepted_count"] == 1
    assert failed_run["rejected_count"] == 0

    # Verify the previous publication pointer was preserved.
    assert pointer_path.read_text(encoding="utf-8") == original_pointer

    # A failed Spark run must not produce a successful manifest.
    assert not (
        tmp_path / "reports" / "run_manifest.json"
    ).exists()
