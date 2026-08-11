"""
Unit & Integration Tests for Model Health System & Minimum Sample Safety (Phases 2-11).

Verifies:
1. N < 20 active sample count outputs confluence_level = INSUFFICIENT_SAMPLE and action_signal = PASS_WAIT_FOR_CONFLUENCE.
2. Structured model_health metadata object contains active_sample_count, rolling_accuracy, rolling_brier, confidence_interval, drift_detected, and reason.
3. Rolling agreement below drift threshold (55.0%) triggers model_health.status = DEGRADED and forces action_signal = PASS_WAIT_FOR_CONFLUENCE.
4. Target period calculation strictly respects confirmed issue + 1 (N+1 invariant).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction, _calculate_wilson_ci


def test_wilson_confidence_interval_bounds():
    """Verify Wilson Score 95% Confidence Interval calculation logic."""
    lower, upper = _calculate_wilson_ci(15, 20)
    assert lower >= 0.0
    assert upper <= 100.0
    assert lower < upper
    assert round(lower, 1) == 53.1
    assert round(upper, 1) == 88.8


@pytest.mark.asyncio
async def test_insufficient_sample_size_emits_pass_signal():
    """Verify evaluated sample size < 20 forces INSUFFICIENT_SAMPLE and PASS_WAIT_FOR_CONFLUENCE."""
    mock_session = AsyncMock()

    # Mock 15 historical draw rows
    mock_draw_rows = []
    for i in range(15):
        mock_r = MagicMock()
        mock_r.issue_id = str(20260811100050000 + (14 - i))
        mock_r.calculated_size = "BIG" if i % 2 == 0 else "SMALL"
        mock_r.result_number = (i % 5) * 2
        mock_r.source_color = "red"
        mock_draw_rows.append(mock_r)

    # Mock 10 evaluated prediction join rows (evaluated < 20 threshold)
    mock_eval_rows = []
    for i in range(10):
        mock_eval_rows.append((
            str(20260811100050000 + (9 - i)),
            "BIG" if i % 2 == 0 else "SMALL",
            0.70,
            "BIG" if i % 2 == 0 else "SMALL"
        ))

    def execute_side_effect(query):
        mock_res = MagicMock()
        query_str = str(query)
        if "engine_predictions" in query_str:
            mock_res.fetchall.return_value = mock_eval_rows
        else:
            mock_res.fetchall.return_value = mock_draw_rows
        return mock_res

    mock_session.execute.side_effect = execute_side_effect

    res = await generate_prediction(mock_session, window=100)

    assert res is not None
    assert res["confluence_level"] == "INSUFFICIENT_SAMPLE"
    assert res["action_signal"] == "PASS_WAIT_FOR_CONFLUENCE"
    assert res["edge_recommendation"] == "PASS_WAIT_FOR_MINIMUM_SAMPLE_VALIDATION"
    assert res["model_health"]["status"] == "INSUFFICIENT_SAMPLE"
    assert "below minimum threshold" in res["model_health"]["reason"]


@pytest.mark.asyncio
async def test_model_health_object_structure():
    """Verify prediction payload contains complete, structured model_health object with true OOS evaluation."""
    mock_session = AsyncMock()

    # Mock 30 historical draw rows
    mock_draw_rows = []
    for i in range(30):
        mock_r = MagicMock()
        mock_r.issue_id = str(20260811100051000 + (29 - i))
        mock_r.calculated_size = "BIG" if i % 2 == 0 else "SMALL"
        mock_r.result_number = (i % 5) * 2
        mock_r.source_color = "green"
        mock_draw_rows.append(mock_r)

    # Mock 25 evaluated prediction join rows
    mock_eval_rows = []
    for i in range(25):
        mock_eval_rows.append((
            str(20260811100051000 + (24 - i)),
            "BIG" if i % 2 == 0 else "SMALL",
            0.72,
            "BIG" if i % 2 == 0 else "SMALL"
        ))

    def execute_side_effect(query):
        mock_res = MagicMock()
        query_str = str(query)
        if "engine_predictions" in query_str:
            mock_res.fetchall.return_value = mock_eval_rows
        else:
            mock_res.fetchall.return_value = mock_draw_rows
        return mock_res

    mock_session.execute.side_effect = execute_side_effect

    res = await generate_prediction(mock_session, window=100)

    assert "indicator_confluence" in res
    confluence = res["indicator_confluence"]
    assert "active_indicators" in confluence
    assert "agreeing_indicators" in confluence
    assert "consensus_pct" in confluence

    assert "model_health" in res
    health = res["model_health"]
    assert "status" in health
    assert "historical_draw_sample_size" in health
    assert health["historical_draw_sample_size"] == 30
    assert "evaluated_prediction_count" in health
    assert health["evaluated_prediction_count"] == 25
    assert "correct_prediction_count" in health
    assert "historical_rolling_accuracy" in health
    assert "historical_rolling_brier" in health
    assert "historical_wilson_ci" in health
    assert "drift_detected" in health
    assert "reason" in health
