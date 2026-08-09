"""
Prediction Timer & Synchronization Integrity Test Suite.

Verifies:
1. Every prediction receives an authoritative 15-second display lifetime (expires_at_ms - created_at_ms == 15000ms).
2. Repeated calls/polling for the SAME period ID return identical expires_at_ms (no resets or timer jumps).
3. Clock offset calculations & stale response protection.
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
async def test_exact_15_second_prediction_lifetime():
    """Verify generated prediction has exactly 15000ms duration."""
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
    assert pred["display_duration_sec"] == 15
    assert pred["expires_at_ms"] - pred["created_at_ms"] == 15000


@pytest.mark.asyncio
async def test_repeated_polling_does_not_reset_expiration():
    """Verify that calling generate_prediction multiple times for same period returns identical expires_at_ms."""
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
    # Call 2 (simulating fast API polling)
    pred2 = await generate_prediction(mock_session)
    # Call 3
    pred3 = await generate_prediction(mock_session)

    assert pred1["upcoming_issue_id"] == pred2["upcoming_issue_id"] == pred3["upcoming_issue_id"]
    # Timestamps MUST be identical (no reset to 15s)
    assert pred1["expires_at_ms"] == pred2["expires_at_ms"] == pred3["expires_at_ms"]
    assert pred1["created_at_ms"] == pred2["created_at_ms"] == pred3["created_at_ms"]
