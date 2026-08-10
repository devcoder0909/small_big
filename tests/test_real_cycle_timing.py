"""
Test suite for 30-Second Cycle Timing Diagnostics.
Verifies pipeline latency markers and ensures target prediction completes within cycle window (< 2000ms total).
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction


@pytest.fixture(autouse=True)
def mock_ai_rotator(monkeypatch):
    async def _mock_ai(*args, **kwargs):
        return None
    monkeypatch.setattr("app.analytics.ai_rotator.fetch_ai_prediction", _mock_ai)


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


@pytest.mark.asyncio
async def test_30_second_cycle_timing():
    mock_session = AsyncMock()
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", str(50000 + 50 - i)) for i in range(50)]
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session.execute.return_value = mock_exec

    t0_result_confirmed = time.monotonic()
    pred = await generate_prediction(mock_session, 50)
    t2_prediction_locked = time.monotonic()

    cycle_latency_ms = (t2_prediction_locked - t0_result_confirmed) * 1000.0

    # Must finish in < 2000ms to allow 28+ seconds for user display
    assert cycle_latency_ms < 2000.0
    assert pred["upcoming_issue_id"] == "50051"
