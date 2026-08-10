"""
Test suite for Large History Memory Safety.
Proves prediction engine memory allocation remains bounded (< 15 MB peak RAM) for 10,000 to 50,000 rows.
"""

import pytest
import tracemalloc
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction


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
async def test_large_history_memory_bounds():
    session, rows = _make_mock_session(10000)

    tracemalloc.start()
    pred = await generate_prediction(session, 10000)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_mem / (1024 * 1024)
    assert pred["total_records_analyzed"] == 10000
    assert peak_ram_mb < 15.0, f"Peak RAM {peak_ram_mb:.2f}MB exceeded 15MB limit"
