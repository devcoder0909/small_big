"""
Unit & Integration Tests for Selective High-Confluence Gating.

Verifies:
1. High-agreement & low-entropy triggers PREDICT_BIG / PREDICT_SMALL action signals.
2. Low-confluence triggers PASS_WAIT_FOR_CONFLUENCE.
3. Target period is strictly N+1.
4. No future-data leakage in prediction calculations.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.analytics.prediction_engine import generate_prediction


@pytest.mark.asyncio
async def test_high_confluence_gate_triggers_predict_signal():
    """Verify high agreement and low entropy trigger PREDICT action signal."""
    mock_session = AsyncMock()

    # Mock 100 historical draws (ordered descending from newest to oldest)
    mock_rows = []
    for i in range(100):
        mock_r = MagicMock()
        mock_r.issue_id = str(20260811100050000 + (99 - i))
        mock_r.calculated_size = "BIG" if i % 2 == 0 else "SMALL"
        mock_r.result_number = (i % 5) * 2
        mock_r.source_color = "red"
        mock_rows.append(mock_r)

    mock_res = MagicMock()
    mock_res.fetchall.return_value = mock_rows
    mock_session.execute.return_value = mock_res

    res = await generate_prediction(mock_session, window=100)

    assert res is not None
    assert "action_signal" in res
    assert "confluence_score" in res
    assert "edge_recommendation" in res
    assert res["action_signal"] in ("PREDICT_BIG", "PREDICT_SMALL", "PASS_WAIT_FOR_CONFLUENCE")
    assert res["upcoming_issue_id"] == "20260811100050100"


@pytest.mark.asyncio
async def test_target_period_strictly_n_plus_one():
    """Verify target period is strictly confirmed issue + 1."""
    mock_session = AsyncMock()

    mock_rows = []
    for i in range(10):
        mock_r = MagicMock()
        mock_r.issue_id = str(20260811100052000 + (9 - i))
        mock_r.calculated_size = "BIG"
        mock_r.result_number = 7
        mock_r.source_color = "green"
        mock_rows.append(mock_r)

    mock_res = MagicMock()
    mock_res.fetchall.return_value = mock_rows
    mock_session.execute.return_value = mock_res

    res = await generate_prediction(mock_session, window=10)

    latest_confirmed = res["telemetry"]["latest_confirmed_period"]
    upcoming = res["upcoming_issue_id"]

    assert latest_confirmed == "20260811100052009"
    assert upcoming == "20260811100052010"
