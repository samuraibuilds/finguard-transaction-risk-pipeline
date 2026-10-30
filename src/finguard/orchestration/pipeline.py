
from pathlib import Path
from datetime import datetime, timezone
import uuid
import json
import logging
import os

from finguard.common import config, logging_setup
from finguard.ingestion.reader import read_csv
from finguard.validation.validator import validate
from finguard.transformations.spark_pipeline import run_spark
from finguard.models.train import train
from finguard.inference.scorer import load_model, score
from finguard.analytics.sql_runner import run as analytics
from finguard.monitoring.monitor import monitor
from finguard.storage.warehouse import run_record
from finguard.storage.publisher import publish_batch


def run_pipeline(config_path="config/config.yaml", train_new=True):
    c = config(config_path)
    logging_setup()

    p = c["paths"]
    rid = str(uuid.uuid4())
    batch_id = c["project"]["batch_id"]
    started = datetime.now(timezone.utc).isoformat()

    raw_n = good_n = bad_n = 0
    status = "failed"
    error = None

    try:
        # 1. Ingest source data
        raw = read_csv(p["input_csv"])
        raw_n = len(raw)

        # 2. Validate and quarantine records
        good, bad, summary = validate(raw, c["validation"])

        good_n = len(good)
        bad_n = len(bad)

        if good.empty:
            raise ValueError("No valid records available for processing")

        if summary["rejection_rate"] > c["validation"]["max_rejection_rate"]:
            raise ValueError("Rejection rate exceeds configured maximum")

        quarantine_dir = Path(p["quarantine_dir"])
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        bad.to_csv(
            quarantine_dir / "rejected_records.csv",
            index=False,
        )

        # 3. Save validated batch
        processed_dir = Path(p["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        tmp = processed_dir / "validated_batch.csv"
        good.to_csv(tmp, index=False)

        # 4. Execute Spark transformations
        spark_stats = run_spark(
            str(tmp),
            p["processed_dir"],
            c["spark"]["app_name"],
            c["spark"]["shuffle_partitions"],
        )

        # Verify Spark row count
        spark_rows = spark_stats.get("spark_rows")

        if spark_rows != len(good):
            raise RuntimeError(
                f"Spark row-count mismatch: expected {len(good)}, "
                f"got {spark_rows}"
            )

        # Verify Spark Parquet output
        spark_output = Path(spark_stats["transactions"])
        parquet_files = list(spark_output.rglob("*.parquet"))

        if not parquet_files:
            raise RuntimeError(
                "Spark transaction Parquet output is missing"
            )

        # 5. Save validated transactions
        validated_parquet = (
            processed_dir / "validated_transactions.parquet"
        )

        good.to_parquet(validated_parquet, index=False)

        if (
            not validated_parquet.is_file()
            or validated_parquet.stat().st_size == 0
        ):
            raise RuntimeError(
                "Validated transaction Parquet output is missing or empty"
            )

        # 6. SQL analytics
        analytics(good, p["reports_dir"])

        # 7. Train or load ML model
        if train_new:
            train(
                good,
                p["model_dir"],
                c["model"]["version"],
                c["project"]["random_seed"],
            )

        model, meta = load_model(
            p["model_dir"],
            c["model"]["version"],
        )

        # 8. Batch inference
        scored = score(
            good,
            model,
            meta,
            batch_id,
            c["model"]["high_risk_threshold"],
        )

        # 9. Verify scoring integrity
        required_score_columns = {
            "transaction_id",
            "risk_category",
        }

        missing_score_columns = (
            required_score_columns - set(scored.columns)
        )

        if missing_score_columns:
            raise RuntimeError(
                "Scoring output missing columns: "
                f"{sorted(missing_score_columns)}"
            )

        if len(scored) != len(good):
            raise RuntimeError(
                f"Scoring row-count mismatch: expected {len(good)}, "
                f"got {len(scored)}"
            )

        if scored["transaction_id"].isna().any():
            raise RuntimeError(
                "Scoring output contains missing transaction IDs"
            )

        if scored["transaction_id"].duplicated().any():
            raise RuntimeError(
                "Scoring output contains duplicate transaction IDs"
            )

        if set(scored["transaction_id"]) != set(good["transaction_id"]):
            raise RuntimeError(
                "Scoring output transaction IDs do not match validated input"
            )

        # 10. Monitoring
        monitor(
            good,
            bad,
            scored,
            p["monitoring_dir"],
        )

        # 11. Publish verified batch outputs
        publication = publish_batch(
            scored=scored,
            batch_id=batch_id,
            run_id=rid,
            scored_dir=p["scored_dir"],
        )

        logging.info(
            "Batch published successfully: %s",
            publication["directory"],
        )

        # 12. Build complete run manifest
        finished = datetime.now(timezone.utc).isoformat()

        manifest = {
            "run_id": rid,
            "batch_id": batch_id,
            "status": "success",
            "started_at": started,
            "finished_at": finished,
            "counts": {
                "input": raw_n,
                "accepted": good_n,
                "rejected": bad_n,
                "scored": len(scored),
            },
            "validation": summary,
            "spark": spark_stats,
            "model": {
                "model_version": meta["model_version"],
            },
            "publication": {
                "status": "published",
                "directory": publication["directory"],
                "scored_file": publication["scored_file"],
                "investigation_queue_file": publication[
                    "investigation_queue_file"
                ],
                "high_risk_count": publication["high_risk_count"],
                "published_at": publication["published_at"],
            },
        }

        # 13. Persist manifest using atomic file replacement
        reports_dir = Path(p["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = reports_dir / "run_manifest.json"
        temp_manifest = reports_dir / f".run_manifest-{rid}.tmp"

        try:
            with temp_manifest.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_manifest, manifest_path)

        finally:
            if temp_manifest.exists():
                temp_manifest.unlink()

        if (
            not manifest_path.is_file()
            or manifest_path.stat().st_size == 0
        ):
            raise RuntimeError("Run manifest missing or empty")

        # 14. Mark successful execution
        status = "success"

        logging.info(
            "Pipeline succeeded run_id=%s",
            rid,
        )

        return manifest

    except Exception as e:
        error = str(e)

        logging.exception(
            "Pipeline failed run_id=%s",
            rid,
        )

        raise

    finally:
        run_record(
            p["sqlite_path"],
            {
                "run_id": rid,
                "batch_id": batch_id,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "input_count": raw_n,
                "accepted_count": good_n,
                "rejected_count": bad_n,
                "error": error,
            },
        )
