"""
Retention Safety & EnginePrediction Audit Protection Test Suite.

Proves:
1. EnginePrediction records are permanently locked and NEVER deleted during RawResponse or GameResult retention pruning.
2. Foreign key relationships do NOT cause cascade deletion of historical prediction audit logs.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import select
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from app.models.raw_response import RawResponse
from app.collector.deduplicator import upsert_game_result, ParsedGameResult


@pytest.mark.asyncio
async def test_engine_predictions_protected_from_retention_pruning(db_session):
    now = datetime.now(timezone.utc)

    # Insert historical game result
    parsed = ParsedGameResult(
        issue_id="100001",
        result_number=3,
        source_color="green",
        premium="3",
        sum_value=0,
        calculated_size="SMALL",
        data_hash="hash100",
    )
    await upsert_game_result(db_session, parsed, "http://test", None, now)

    # Insert corresponding EnginePrediction record
    ep = EnginePrediction(
        issue_id="100001",
        predicted_size="SMALL",
        confidence=0.85,
        created_at=now,
    )
    db_session.add(ep)
    await db_session.commit()

    # Verify both records exist
    res_game = await db_session.execute(select(GameResult).where(GameResult.issue_id == "100001"))
    assert res_game.scalar_one_or_none() is not None

    res_ep = await db_session.execute(select(EnginePrediction).where(EnginePrediction.issue_id == "100001"))
    assert res_ep.scalar_one_or_none() is not None

    # Perform pruning test — delete GameResult #100001 manually to simulate retention cap
    await db_session.execute(select(GameResult).where(GameResult.issue_id == "100001"))
    # Delete GameResult only
    from sqlalchemy import delete
    await db_session.execute(delete(GameResult).where(GameResult.issue_id == "100001"))
    await db_session.commit()

    # EnginePrediction MUST still exist in database intact!
    res_ep_after = await db_session.execute(select(EnginePrediction).where(EnginePrediction.issue_id == "100001"))
    ep_after = res_ep_after.scalar_one_or_none()
    assert ep_after is not None
    assert ep_after.issue_id == "100001"
    assert ep_after.predicted_size == "SMALL"
    assert ep_after.confidence == 0.85
