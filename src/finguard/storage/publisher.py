
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def _validate_identifier(value, name):
    """Prevent unsafe directory names."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", value
    ):
        raise ValueError(f"Invalid {name}: {value!r}")


def publish_batch(scored, batch_id, run_id, scored_dir):
    """
    Publish immutable batch artifacts and atomically update current.json.

    A failed staging operation does not change the previous current.json.
    """
    import pandas as pd

    _validate_identifier(batch_id, "batch_id")
    _validate_identifier(run_id, "run_id")

    required = {"transaction_id", "risk_category"}

    if not required.issubset(scored.columns):
        raise ValueError(
            f"Scoring output missing columns: "
            f"{sorted(required - set(scored.columns))}"
        )

    if scored.empty:
        raise ValueError("Cannot publish an empty scoring batch")

    if scored["transaction_id"].isna().any():
        raise ValueError("Missing transaction IDs in scoring output")

    if scored["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in scoring output")

    root = Path(scored_dir)
    batch_root = root / "batches" / batch_id

    batch_root.mkdir(parents=True, exist_ok=True)

    stage = batch_root / f".staging-{run_id}"
    final = batch_root / run_id

    if stage.exists() or final.exists():
        raise FileExistsError(
            f"Publication directory already exists for run {run_id}"
        )

    stage.mkdir()

    try:
        scored_file = stage / "scored.parquet"
        queue_file = stage / "investigation_queue.csv"

        high_risk = scored.loc[
            scored["risk_category"].astype(str).str.lower() == "high"
        ]

        scored.to_parquet(scored_file, index=False)
        high_risk.to_csv(queue_file, index=False)

        # Verify staged artifacts before publication.
        if not scored_file.is_file() or scored_file.stat().st_size == 0:
            raise RuntimeError("Staged Parquet file is missing or empty")

        if not queue_file.is_file():
            raise RuntimeError("Staged investigation queue is missing")

        # Verify the Parquet file can actually be read.
        check = pd.read_parquet(scored_file)

        if len(check) != len(scored):
            raise RuntimeError("Staged Parquet row-count mismatch")

        manifest = {
            "batch_id": batch_id,
            "run_id": run_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(scored),
            "high_risk_count": len(high_risk),
            "scored_file": "scored.parquet",
            "investigation_queue_file": "investigation_queue.csv",
        }

        (stage / "batch_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        # Publish the complete immutable directory.
        # Both paths are on the same filesystem.
        stage.rename(final)

        # Update the pointer only after the directory is published.
        pointer = {
            **manifest,
            "directory": str(final.relative_to(root)),
            "scored_file": str(
                (final / "scored.parquet").relative_to(root)
            ),
            "investigation_queue_file": str(
                (final / "investigation_queue.csv").relative_to(root)
            ),
        }

        pointer_path = root / "current.json"
        temp_pointer = root / f".current-{run_id}.tmp"

        try:
            with temp_pointer.open("w", encoding="utf-8") as f:
                json.dump(pointer, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replacement of the pointer file on the same volume.
            os.replace(temp_pointer, pointer_path)

        finally:
            if temp_pointer.exists():
                temp_pointer.unlink()

        return pointer

    except Exception:
        # Remove incomplete staging data.
        # Never delete a previously published batch.
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)

        raise
