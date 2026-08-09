"""Tests for gap detection and recovery."""

import pytest
from app.services.recovery_service import detect_gaps
from app.models.game_result import GameResult
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_detect_gaps(db_session):
    """Test missing sequential issue detection."""
    now = datetime.now(timezone.utc)
    # Insert records with gap: 1001, 1002, 1005 (missing 1003, 1004)
    records = [
        GameResult(issue_id="1005", result_number=7, source_color="green", calculated_size="BIG", first_observed_at=now, last_observed_at=now, source_url="test"),
        GameResult(issue_id="1002", result_number=3, source_color="red", calculated_size="SMALL", first_observed_at=now, last_observed_at=now, source_url="test"),
        GameResult(issue_id="1001", result_number=1, source_color="green", calculated_size="SMALL", first_observed_at=now, last_observed_at=now, source_url="test"),
    ]
    for r in records:
        db_session.add(r)
    await db_session.commit()

    gaps = await detect_gaps(db_session)
    assert len(gaps) == 1
    assert gaps[0]["missing_count"] == 2
    assert "1003" in gaps[0]["missing_ids"]
    assert "1004" in gaps[0]["missing_ids"]
