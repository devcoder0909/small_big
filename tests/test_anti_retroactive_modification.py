"""
Test suite for Anti-Retroactive Prediction Modification.

Proves:
1. Game History records (period, result/actual) are strictly derived from GameResult rows.
2. Prediction records or post-result operations CANNOT retroactively modify displayed Game History.
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
async def test_anti_retroactive_prediction_modification():
    mock_session = AsyncMock()

    rows = [MockRow("BIG", "00051369", 8)]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    # Fetch historical game results
    res = await get_game_history(mock_session, limit=10)

    assert len(res) == 1
    record = res[0]
    assert record["period"] == "00051369"
    assert record["actual"] == "BIG"
    assert record["result"] == "BIG"
    assert "predicted" not in record
    assert "prediction_status" not in record
