"""Tests for deduplication logic."""

import pytest
from datetime import datetime, timezone
from app.collector.deduplicator import upsert_game_result, get_total_record_count
from app.collector.parser import ParsedGameResult


@pytest.mark.asyncio
async def test_deduplication_insert_and_update(db_session):
    """Test that duplicate insertions do not create duplicate rows."""
    now = datetime.now(timezone.utc)
    parsed = ParsedGameResult(
        issue_id="20260809100051311",
        result_number=7,
        source_color="green",
        premium="7",
        sum_value=0,
        calculated_size="BIG",
        data_hash="hash123",
    )

    # First insert
    is_new1, status1 = await upsert_game_result(
        db_session, parsed, "http://test", None, now
    )
    assert is_new1 is True
    assert status1 == "NEW_RECORD_DETECTED"

    total1 = await get_total_record_count(db_session)
    assert total1 == 1

    # Second insert with same issue_id (duplicate)
    is_new2, status2 = await upsert_game_result(
        db_session, parsed, "http://test", None, now
    )
    assert is_new2 is False
    assert status2 == "DUPLICATE_SKIPPED"

    total2 = await get_total_record_count(db_session)
    assert total2 == 1  # Record count MUST remain 1!
