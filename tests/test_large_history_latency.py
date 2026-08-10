"""
Test suite for Large History Latency Safety.
Verifies total prediction execution time remains safely under 500ms for 10,000 historical records.
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


def _make_mock_session(record_count: int):
    mock_session = AsyncMock()
    rows = []
    for i in range(record_count):
        issue_id = str(100000 + record_count - 1 - i)
        size = "BIG" if (i % 2 == 0) else "SMALL"
        rows.append(MockRow(size, issue_id, number=(i % 10)))

    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session.execute.return_value = mock_exec
    return mock_session, rows


@pytest.mark.asyncio
async def test_large_history_latency_bounds():
    session, rows = _make_mock_session(10000)

    t0 = time.monotonic()
    pred = await generate_prediction(session, 10000)
    t1 = time.monotonic()

    latency_ms = (t1 - t0) * 1000.0
    assert pred["total_records_analyzed"] == 10000
    assert latency_ms < 500.0, f"Latency {latency_ms:.2f}ms exceeded 500ms target limit"
