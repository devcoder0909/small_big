"""
Test suite for Production Telemetry Breakdown.
Verifies engine attaches sanitized telemetry structure (database_ms, feature_extraction_ms, edge_level, selected_window).
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


@pytest.mark.asyncio
async def test_production_telemetry_fields():
    mock_session = AsyncMock()
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", str(50000 + 100 - i)) for i in range(100)]
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session)

    assert "telemetry" in pred
    telemetry = pred["telemetry"]
    assert "latest_confirmed_period" in telemetry
    assert "target_period" in telemetry
    assert "rows_loaded" in telemetry
    assert "selected_window" in telemetry
    assert "regime" in telemetry
    assert "edge_level" in telemetry
    assert "latency_ms" in telemetry
    assert "database_ms" in telemetry["latency_ms"]
    assert "feature_extraction_ms" in telemetry["latency_ms"]
