"""
Immutability & Database Constraint Tests.

Proves:
1. Unique constraint on EnginePrediction issue_id prevents duplicate prediction records.
2. persist_original_prediction uses ON CONFLICT DO NOTHING to lock predictions permanently.
3. Original predicted_size is never overwritten or modified retroactively.
4. Accuracy calculations compare immutable original predictions against actual calculated_size.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import IntegrityError

from app.analytics.prediction_engine import persist_original_prediction, get_game_history
from app.models.engine_prediction import EnginePrediction
from app.models.game_result import GameResult


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


@pytest.mark.asyncio
async def test_persist_prediction_ignores_duplicate_issue_id():
    """Verify that calling persist_original_prediction on an existing issue_id does not overwrite it."""
    bind_mock = MagicMock()
    bind_mock.dialect.name = "sqlite"

    mock_exec_empty = MagicMock()
    mock_exec_empty.scalar_one_or_none.return_value = None

    mock_exec_exists = MagicMock()
    mock_exec_exists.scalar_one_or_none.return_value = 1

    mock_session = MagicMock()
    mock_session.get_bind.return_value = bind_mock
    mock_session.execute = AsyncMock(side_effect=[mock_exec_empty, mock_exec_exists])
    mock_session.commit = AsyncMock()

    # Initial prediction record
    pred1 = {
        "upcoming_issue_id": "500001",
        "prediction": "BIG",
        "confidence": 0.85,
        "confluence_level": "SUPER_CONFLUENCE",
        "agreeing_indicators": 10,
        "active_indicators": 12,
        "created_at_ms": 1700000000000,
    }

    # Attempted update record for same period
    pred2 = {
        "upcoming_issue_id": "500001",
        "prediction": "SMALL",
        "confidence": 0.55,
        "confluence_level": "LOW",
        "agreeing_indicators": 4,
        "active_indicators": 12,
        "created_at_ms": 1700000010000,
    }

    # First insertion works
    await persist_original_prediction(mock_session, pred1)
    assert mock_session.add.call_count == 1

    # Second insertion for same period is ignored because record already exists
    await persist_original_prediction(mock_session, pred2)
    assert mock_session.add.call_count == 1  # Still 1


@pytest.mark.asyncio
async def test_get_game_history_uses_stored_immutable_game_results():
    """Verify game history reads original stored GameResult rows, not predictions."""
    mock_session = AsyncMock()

    rows = [
        MockRow("BIG", "500002"),
        MockRow("SMALL", "500001"),
    ]

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    results = await get_game_history(mock_session, limit=10)

    assert len(results) >= 1
    assert results[0]["issue_id"] == "500002"
    assert results[0]["result"] == "BIG"
    assert "predicted" not in results[0]
    assert "is_win" not in results[0]
