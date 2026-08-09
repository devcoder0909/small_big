"""Tests for analytics engine."""

import pytest
from app.analytics.streaks import calculate_streaks
from app.analytics.frequency import calculate_frequency
from app.analytics.prediction_engine import generate_prediction
from app.models.game_result import GameResult
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_analytics_on_empty_db(db_session):
    """Test analytics return safe defaults when database is empty."""
    freq = await calculate_frequency(db_session, 100)
    assert freq["total"] == 0
    assert freq["small_count"] == 0

    prediction = await generate_prediction(db_session)
    assert prediction["prediction"] is None
    assert prediction["status"] == "INSUFFICIENT_DATA"


@pytest.mark.asyncio
async def test_streak_analytics(db_session):
    """Test streak calculation with mock records."""
    now = datetime.now(timezone.utc)
    records = [
        GameResult(issue_id="103", result_number=7, source_color="green", calculated_size="BIG", first_observed_at=now, last_observed_at=now, source_url="test"),
        GameResult(issue_id="102", result_number=8, source_color="red", calculated_size="BIG", first_observed_at=now, last_observed_at=now, source_url="test"),
        GameResult(issue_id="101", result_number=1, source_color="green", calculated_size="SMALL", first_observed_at=now, last_observed_at=now, source_url="test"),
    ]
    for r in records:
        db_session.add(r)
    await db_session.commit()

    streaks = await calculate_streaks(db_session)
    assert streaks["current_size"] == "BIG"
    assert streaks["current_streak"] == 2
