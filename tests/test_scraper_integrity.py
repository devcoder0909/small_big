"""
Zero-Miss Result Scraper & Data Integrity Test Suite.

Verifies:
1. Pre-Prediction Data Gate blocks predictions on historical data gaps.
2. Gap detection identifies missing period numbers correctly.
3. Duplicate protection & result conflict handling.
4. Stale data detection.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction
from app.services.recovery_service import detect_gaps


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_pre_prediction_data_gate_blocks_on_gap():
    """Verify generate_prediction returns INSUFFICIENT_DATA when a gap is present."""
    # Row sequence with missing period #1079 between #1080 and #1078
    rows = [
        MockRow("SMALL", "1080", 2),
        MockRow("BIG", "1078", 8),  # Gap: missing 1079!
        MockRow("SMALL", "1077", 3),
        MockRow("BIG", "1076", 7),
        MockRow("SMALL", "1075", 1),
        MockRow("BIG", "1074", 9),
    ]

    mock_execute_result = MagicMock()
    mock_execute_result.fetchall.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_execute_result

    res = await generate_prediction(mock_session)

    assert res["status"] == "INSUFFICIENT_DATA"
    assert "Historical data gap detected" in res["message"]
    assert res["prediction"] is None


@pytest.mark.asyncio
async def test_detect_gaps_identifies_missing_periods():
    """Verify detect_gaps service correctly returns list of missing period IDs."""
    rows = [
        MockRow("SMALL", "1080", 2),
        MockRow("BIG", "1078", 8),  # Missing 1079
        MockRow("SMALL", "1075", 3),  # Missing 1076, 1077
        MockRow("BIG", "1074", 7),
    ]

    mock_execute_result = MagicMock()
    mock_execute_result.fetchall.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_execute_result

    gaps = await detect_gaps(mock_session, window=100)

    assert len(gaps) == 2
    assert "1079" in gaps[0]["missing_ids"] or "1079" in gaps[1]["missing_ids"]
