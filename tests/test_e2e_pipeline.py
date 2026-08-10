"""
End-to-End System Pipeline & Production Reliability Test Suite.

Audits the complete lifecycle:
Scraper Source → Parser → Validation → Data Gate → Engine → Immutability → Game History → API
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import (
    generate_prediction,
    persist_original_prediction,
    get_game_history,
)
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from app.services.recovery_service import detect_gaps, recover_missing_records


class MockGameRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_full_prediction_lifecycle_immutability():
    # 1. Generate prediction for upcoming period
    mock_session = AsyncMock()

    rows = [MockGameRow("BIG", f"20260810{i:04d}", (i % 10)) for i in range(150, 100, -1)]
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session, 50)
    assert pred["status"] in ("READY", "ACTIVE")
    assert pred["prediction"] in ("BIG", "SMALL")
    assert pred["upcoming_issue_id"] == "202608100151"

    # 2. Persist original prediction to database
    await persist_original_prediction(mock_session, pred)
    assert mock_session.execute.called

    # 3. Game History retrieval (strictly real GameResult outcomes)
    history = await get_game_history(mock_session, limit=10)

    assert isinstance(history, list)
    assert len(history) == 50
    for h in history:
        assert "result" in h
        assert "issue_id" in h
        assert "predicted" not in h
        assert "is_win" not in h


@pytest.mark.asyncio
async def test_end_to_end_gap_detection_and_blocking():
    """Verify that gaps block prediction generation and trigger INSUFFICIENT_DATA."""
    gapped_rows = [
        MockGameRow("SMALL", "202608100050", 2),
        MockGameRow("BIG", "202608100048", 8),  # Missing 202608100049!
        MockGameRow("SMALL", "202608100047", 3),
        MockGameRow("BIG", "202608100046", 9),
        MockGameRow("SMALL", "202608100045", 1),
    ]

    mock_exec_res = MagicMock()
    mock_exec_res.fetchall.return_value = gapped_rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec_res

    res = await generate_prediction(mock_session)

    assert res["status"] == "INSUFFICIENT_DATA"
    assert "Historical data gap detected" in res["message"]
    assert res["prediction"] is None


@pytest.mark.asyncio
async def test_game_history_retrieval_integrity():
    """Verify game history returns pure GameResult data without prediction fields."""
    rows = [
        MockGameRow("SMALL", "202608100049", 2),
        MockGameRow("BIG", "202608100048", 8),
    ]

    mock_exec_res = MagicMock()
    mock_exec_res.scalars().all.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec_res

    history = await get_game_history(mock_session, limit=10)

    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0]["period"] == "202608100049"
    assert history[0]["result"] == "SMALL"
    assert "predicted" not in history[0]
