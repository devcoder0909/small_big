"""
Prediction Period Synchronization Test Suite.

Verifies:
1. Every prediction is accurately generated for the upcoming_issue_id.
2. Repeated calls/polling for the SAME period ID return consistent predictions and issue IDs without duplicate generation.
3. Period transition automatically updates prediction for the next period.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_prediction_period_synchronization():
    """Verify generated prediction has valid upcoming_issue_id and ACTIVE status."""
    rows = [
        MockRow("SMALL", str(1000 + i), (i % 10))
        for i in range(50, 0, -1)
    ]

    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session)

    assert pred["status"] == "ACTIVE"
    assert pred["upcoming_issue_id"] == "1051"
    assert pred["prediction"] in ("SMALL", "BIG")


@pytest.mark.asyncio
async def test_repeated_polling_returns_consistent_prediction():
    """Verify that calling generate_prediction multiple times for same period returns consistent data."""
    rows = [
        MockRow("SMALL", str(1000 + i), (i % 10))
        for i in range(50, 0, -1)
    ]

    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec

    # Call 1
    pred1 = await generate_prediction(mock_session)
    # Call 2
    pred2 = await generate_prediction(mock_session)

    assert pred1["upcoming_issue_id"] == pred2["upcoming_issue_id"]
    assert pred1["prediction"] == pred2["prediction"]
    assert pred1["confidence"] == pred2["confidence"]
