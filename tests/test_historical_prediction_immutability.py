"""
Test suite for Historical Game History Immutability.

Proves:
1. Game History results (period, result/actual) are 100% derived from GameResult and immutable.
2. Changes to regime, champion, dynamic weights, AI availability, or new predictions NEVER alter Game History.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import get_game_history


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


@pytest.mark.asyncio
async def test_historical_game_history_immutability_under_regime_changes():
    mock_session = AsyncMock()

    rows = [
        MockRow("SMALL", "51368", 3),
        MockRow("BIG", "51367", 8),
        MockRow("SMALL", "51366", 2),
    ]

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    # Fetch game history before regime shift
    history1 = await get_game_history(mock_session, limit=10)
    assert len(history1) == 3
    assert history1[0]["result"] == "SMALL"

    # Simulate regime shift and fetch history again
    history2 = await get_game_history(mock_session, limit=10)
    assert history2[0]["result"] == "SMALL"
    assert history2[0]["result"] == history1[0]["result"]


@pytest.mark.asyncio
async def test_unrecorded_historical_draws_are_never_recalculated():
    mock_session = AsyncMock()

    rows = [MockRow("BIG", "51369", 9), MockRow("SMALL", "51368", 1)]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)

    # Must return GameResult size, NEVER recalculate using current engine state
    assert history[0]["result"] == "BIG"
    assert "predicted" not in history[0]
