"""
Unit tests for safe database reset script (scripts/reset_game_data.py).
"""

import pytest
import pytest_asyncio
from sqlalchemy import select, func

from app.collector.deduplicator import upsert_batch
from app.collector.parser import ParsedGameResult
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from scripts.reset_game_data import perform_reset, CONFIRMATION_PHRASE


@pytest.mark.asyncio
async def test_reset_game_data_script_confirmation(db_session):
    """Verify reset script rejects invalid confirmation phrase."""
    with pytest.raises(SystemExit):
        await perform_reset("INVALID_CONFIRMATION_PHRASE")


@pytest.mark.asyncio
async def test_reset_game_data_script_dry_run(db_session):
    """Verify dry run mode leaves records untouched."""
    session = db_session
    item = ParsedGameResult("20260809100099999", 5, "green", None, None, "BIG", "hash_reset_test")
    await upsert_batch(session, [item], "http://test", None, None)
    await session.commit()

    # Dry run
    await perform_reset(CONFIRMATION_PHRASE, dry_run=True)

    # Verify count is still 1
    count = (await session.execute(select(func.count()).select_from(GameResult))).scalar()
    assert count == 1
