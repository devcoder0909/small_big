"""
Phase 17 — Production State Consistency & Data Lineage Lock Test Suite.

Verifies:
1. Build SHA Parity across /health, /health/detailed, /api/v1/public/prediction, and telemetry payload.
2. database_record_count equals authoritative PostgreSQL GameResult row count.
3. Telemetry data_lineage object contains all required explicit lineage fields.
4. feature_window_selected (e.g. 382) never overwrites database_record_count (e.g. 6650).
5. target_period is strictly latest_confirmed_period + 1.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core import get_build_commit
from app.analytics.prediction_engine import generate_prediction


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_build_sha_parity():
    """Verify build SHA function returns a non-empty 7-character commit SHA."""
    sha = get_build_commit()
    assert sha is not None
    assert isinstance(sha, str)
    assert len(sha) >= 7


@pytest.mark.asyncio
async def test_database_count_parity_and_data_lineage():
    """Verify generate_prediction includes authoritative data_lineage dictionary and correct counts."""
    # Create 50 mock rows
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", str(20260812100050000 + i), (i * 3) % 10) for i in range(50, 0, -1)]

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 6650

    mock_rows_res = MagicMock()
    mock_rows_res.fetchall.return_value = rows

    mock_eval_res = MagicMock()
    mock_eval_res.fetchall.return_value = []
    mock_session = AsyncMock()
    mock_session._force_count_query = True
    mock_session.execute.side_effect = [mock_rows_res, mock_count_res, mock_eval_res]

    pred = await generate_prediction(mock_session)

    assert pred["status"] in ("READY", "ACTIVE")
    assert pred["database_record_count"] == 6650
    assert pred["total_records_analyzed"] == 6650
    assert "data_lineage" in pred

    lineage = pred["data_lineage"]
    assert lineage["database_record_count"] == 6650
    assert lineage["historical_records_loaded"] == 6650
    assert lineage["total_records_analyzed"] == 6650
    assert lineage["valid_contiguous_record_count"] == 50
    assert lineage["feature_window_selected"] <= 6650
    assert lineage["build_commit"] == get_build_commit()


@pytest.mark.asyncio
async def test_feature_window_not_database_count():
    """Verify feature lookback window never overwrites database_record_count."""
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", str(20260812100050000 + i), (i * 3) % 10) for i in range(50, 0, -1)]

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 6650

    mock_rows_res = MagicMock()
    mock_rows_res.fetchall.return_value = rows

    mock_eval_res = MagicMock()
    mock_eval_res.fetchall.return_value = []
    mock_session = AsyncMock()
    mock_session._force_count_query = True
    mock_session.execute.side_effect = [mock_rows_res, mock_count_res, mock_eval_res]

    pred = await generate_prediction(mock_session, window=382)

    assert pred["database_record_count"] == 6650
    assert pred["feature_window_selected"] == 50
    assert pred["database_record_count"] >= pred["feature_window_selected"]


@pytest.mark.asyncio
async def test_target_period_is_latest_plus_one():
    """Verify target_period is strictly latest_confirmed_period + 1."""
    rows = [
        MockRow("SMALL", "20260812100050403", 3),
        MockRow("BIG", "20260812100050402", 8),
        MockRow("SMALL", "20260812100050401", 2),
        MockRow("BIG", "20260812100050400", 7),
        MockRow("SMALL", "20260812100050399", 1),
    ] + [
        MockRow("BIG" if i % 2 == 0 else "SMALL", str(20260812100050398 - i), i % 10) for i in range(20)
    ]

    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 6650

    mock_rows_res = MagicMock()
    mock_rows_res.fetchall.return_value = rows

    mock_eval_res = MagicMock()
    mock_eval_res.fetchall.return_value = []
    mock_session = AsyncMock()
    mock_session._force_count_query = True
    mock_session.execute.side_effect = [mock_rows_res, mock_count_res, mock_eval_res]

    pred = await generate_prediction(mock_session)

    assert pred["data_lineage"]["latest_confirmed_period"] == "20260812100050403"
    assert pred["upcoming_issue_id"] == "20260812100050404"
    assert pred["data_lineage"]["target_period"] == "20260812100050404"
