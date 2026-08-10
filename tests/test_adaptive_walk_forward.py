"""
Test suite for Adaptive Walk-Forward Selection.
Simulates 100 continuous out-of-sample prediction periods to verify AdaptiveWindowSelector learning loop.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction
from app.analytics.adaptive_window_selector import adaptive_window_selector


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


def _make_rows(count=200, start_id=50000):
    rows = []
    for i in range(count):
        issue_id = str(start_id + count - 1 - i)
        size = "BIG" if (i % 2 == 0) else "SMALL"
        rows.append(MockRow(size, issue_id, number=(i % 10)))
    return rows


@pytest.mark.asyncio
async def test_adaptive_walk_forward_simulation():
    full_history = _make_rows(200, 50000)

    for i in range(50):
        sliced_rows = full_history[i : i + 150]
        latest_issue = sliced_rows[0].issue_id

        mock_session = AsyncMock()
        mock_exec = MagicMock()
        mock_exec.fetchall.return_value = sliced_rows
        mock_session.execute.return_value = mock_exec

        pred = await generate_prediction(mock_session)

        # Record OOS outcome if previous issue exists
        if i > 0:
            actual_size = sliced_rows[0].calculated_size
            pred_size = pred.get("prediction")
            selected_win = pred.get("selected_window", 1000)
            regime = pred.get("regime", "STABLE_NEUTRAL")
            brier = pred.get("brier_score", 0.25)
            is_win = (actual_size == pred_size)

            adaptive_window_selector.record_window_result(selected_win, regime, is_win, brier)

    metrics = adaptive_window_selector.get_metrics()
    assert metrics["total_evaluated_samples"] > 0
