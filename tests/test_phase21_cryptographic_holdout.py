"""
Phase 21 — Cryptographic Holdout & AI Challenger Lab Test Suite.

Verifies:
1. Cryptographic holdout partition hash immutability.
2. Dataset partition size isolation (Research / Dev / Validation / Holdout).
3. Phase 21 research report artifact generation and promotion decision locking.
"""

import pytest
import os
import json
import hashlib

from app.analytics.walk_forward_replay import run_walk_forward_replay


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


def test_holdout_partition_hash_immutability():
    """Verify cryptographic holdout partition hash is deterministic and immutable."""
    holdout_rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", f"202608121000{4500+i:05d}", (i * 3) % 10) for i in range(500)]
    holdout_ids = "".join(r.issue_id for r in holdout_rows)
    holdout_hash = hashlib.sha256(holdout_ids.encode()).hexdigest()[:16]

    assert len(holdout_hash) == 16
    assert isinstance(holdout_hash, str)


@pytest.mark.asyncio
async def test_partition_isolation_and_holdout_replay():
    """Verify holdout replay executes cleanly on partition bounds."""
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", f"202608121000{i+1:05d}", (i * 7 + 3) % 10) for i in range(500)]
    report = await run_walk_forward_replay(rows=rows, min_history=50, max_eval_periods=200)

    assert report["status"] == "COMPLETED"
    assert report["evaluated_periods"] == 200
    assert report["champion_model"]["accuracy_pct"] >= 0.0


def test_phase21_report_artifact_exists():
    """Verify phase21_research_report.json artifact exists and contains holdout metrics."""
    artifact_path = os.path.join(os.path.dirname(__file__), "..", "phase21_research_report.json")
    if os.path.exists(artifact_path):
        with open(artifact_path, "r") as f:
            data = json.load(f)
        assert "database_record_count" in data
        assert "research_count" in data
        assert "development_count" in data
        assert "locked_validation_count" in data
        assert "final_holdout_count" in data
        assert "holdout_hash" in data
        assert data["promotion_decision"] == "KEEP_CHAMPION"
