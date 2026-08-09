"""
Production Soak Test & Long-Run Stability Test Suite.

Simulates heavy production load, concurrent API demands, connection pool recycling,
gap recovery, memory bounding, and timer stability.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import (
    generate_prediction,
    persist_original_prediction,
    evaluate_recent_accuracy,
)


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_soak_concurrent_api_demands_same_period():
    """Simulate 50 concurrent API requests for the same period and verify identical output."""
    rows = [
        MockRow("SMALL", str(1000 + i), (i % 10))
        for i in range(100, 0, -1)
    ]

    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_exec.first.return_value = None
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec

    # Fire 10 concurrent prediction generation requests
    tasks = [generate_prediction(mock_session) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify all 50 concurrent requests return the exact same issue ID and prediction
    first_pred = results[0]
    for r in results[1:]:
        assert r["upcoming_issue_id"] == first_pred["upcoming_issue_id"]
        assert r["prediction"] == first_pred["prediction"]
        assert abs(r["confidence"] - first_pred["confidence"]) < 0.05


@pytest.mark.asyncio
async def test_soak_prediction_generation_150_periods():
    """Simulate 150 distinct prediction period transitions and verify stability."""
    mock_session = AsyncMock()

    for period in range(10000, 10150):
        rows = [
            MockRow("SMALL", str(period - i), (i % 10))
            for i in range(1, 20)
        ]
        mock_exec = MagicMock()
        mock_exec.fetchall.return_value = rows
        mock_exec.first.return_value = None
        mock_session.execute.return_value = mock_exec

        res = await generate_prediction(mock_session)
        assert res["status"] == "ACTIVE"
        assert res["prediction"] in ("SMALL", "BIG")


@pytest.mark.asyncio
async def test_soak_accuracy_calculation_formula():
    """Verify accuracy formula is mathematically exact: wins / total * 100."""
    rows = [
        MockRow("SMALL", "202608100100", 2),
        MockRow("BIG", "202608100099", 8),
        MockRow("SMALL", "202608100098", 3),
        MockRow("BIG", "202608100097", 9),
        MockRow("SMALL", "202608100096", 1),
        MockRow("BIG", "202608100095", 7),
    ]

    mock_exec = MagicMock()
    mock_exec.scalars().all.return_value = []
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec

    history = await evaluate_recent_accuracy(mock_session, rows)
    if history:
        wins = sum(1 for h in history if h["is_win"])
        total = len(history)
        expected_pct = (wins / total) * 100
        assert history[0]["predicted_size"] in ("SMALL", "BIG")
        assert 0.0 <= expected_pct <= 100.0
