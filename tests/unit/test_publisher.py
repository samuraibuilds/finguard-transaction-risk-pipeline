
import json

import pandas as pd
import pytest

from finguard.storage.publisher import publish_batch


def test_failed_publication_preserves_previous_batch(tmp_path, monkeypatch):
    scored_dir = tmp_path / "scored"

    first_batch = pd.DataFrame({
        "transaction_id": ["TX001", "TX002"],
        "risk_category": ["low", "high"],
        "risk_score": [0.1, 0.95],
    })

    # First publication must succeed.
    first = publish_batch(
        scored=first_batch,
        batch_id="demo-batch",
        run_id="run-001",
        scored_dir=scored_dir,
    )

    pointer_path = scored_dir / "current.json"

    assert pointer_path.is_file()

    previous_pointer = pointer_path.read_text(encoding="utf-8")
    pointer_data = json.loads(previous_pointer)

    assert pointer_data["run_id"] == "run-001"
    assert pointer_data["row_count"] == 2

    published_dir = scored_dir / pointer_data["directory"]

    assert (published_dir / "scored.parquet").is_file()
    assert (published_dir / "investigation_queue.csv").is_file()

    # Simulate a failure while writing the next batch.
    def fail_parquet_write(*args, **kwargs):
        raise RuntimeError("Simulated Parquet write failure")

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        fail_parquet_write,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated Parquet write failure",
    ):
        publish_batch(
            scored=first_batch,
            batch_id="demo-batch",
            run_id="run-002",
            scored_dir=scored_dir,
        )

    # The previous pointer must remain unchanged.
    assert pointer_path.read_text(encoding="utf-8") == previous_pointer

    # The previous successful batch must still exist.
    assert (published_dir / "scored.parquet").is_file()

    # Failed staging data must be cleaned up.
    assert not (
        scored_dir / "batches" / "demo-batch" / ".staging-run-002"
    ).exists()
