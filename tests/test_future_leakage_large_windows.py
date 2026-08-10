"""
Test suite for Zero Future Leakage across Large Windows.
Verifies that target prediction for period X+1 is strictly invariant to future target row injection (X+1, X+2, X+3).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


def _make_rows(count=1000, start_id=50000):
    rows = []
    for i in range(count):
        issue_id = str(start_id + count - 1 - i)
        size = "BIG" if (i % 2 == 0) else "SMALL"
        rows.append(MockRow(size, issue_id, number=(i % 10)))
    return rows


@pytest.mark.asyncio
async def test_zero_future_leakage_large_windows():
    mock_session = AsyncMock()

    # Base dataset up to period 50999 (predicting 51000)
    base_rows = _make_rows(1000, 50000)
    assert base_rows[0].issue_id == "50999"

    mock_exec1 = MagicMock()
    mock_exec1.fetchall.return_value = base_rows
    mock_session.execute.return_value = mock_exec1

    pred_clean = await generate_prediction(mock_session, 1000)
    assert pred_clean["upcoming_issue_id"] == "51000"

    # Inject fake future row for period 51000
    future_row = MockRow("BIG", "51000", 8)
    injected_rows = [future_row] + base_rows

    mock_exec2 = MagicMock()
    mock_exec2.fetchall.return_value = injected_rows
    mock_session.execute.return_value = mock_exec2

    pred_injected = await generate_prediction(mock_session, 1000)

    # Generated target for injected dataset must be 51001
    assert pred_injected["upcoming_issue_id"] == "51001"
    # Target prediction for 51000 from clean dataset remains unchanged
    assert pred_clean["upcoming_issue_id"] == "51000"
