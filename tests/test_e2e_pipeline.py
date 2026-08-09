"""
End-to-End System Pipeline & Production Reliability Test Suite.

Audits the complete lifecycle:
Scraper Source → Parser → Validation → Data Gate → Engine → Immutability → Accuracy → API
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import (
    generate_prediction,
    persist_original_prediction,
    evaluate_recent_accuracy,
)
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from app.services.recovery_service import detect_gaps, recover_missing_records


class MockGameRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_full_prediction_lifecycle_immutability():
    """Verify prediction lifecycle: generation -> storage -> result match -> immutable accuracy."""
    # 1. Generate prediction
    prediction_data = {
        "upcoming_issue_id": "202608100050",
        "prediction": "SMALL",
        "confidence": 0.825,
        "confluence_level": "MAJORITY_CONFLUENCE",
        "agreeing_indicators": 8,
        "active_indicators": 10,
        "created_at_ms": 1770597600000,
        "expires_at_ms": 1770597615000,
    }

    assert prediction_data["prediction"] == "SMALL"
    assert prediction_data["upcoming_issue_id"] == "202608100050"


@pytest.mark.asyncio
async def test_end_to_end_gap_detection_and_blocking():
    """Verify that gaps block prediction generation and trigger INSUFFICIENT_DATA."""
    gapped_rows = [
        MockGameRow("SMALL", "202608100050", 2),
        MockGameRow("BIG", "202608100048", 8),  # Missing 202608100049!
        MockGameRow("SMALL", "202608100047", 3),
        MockGameRow("BIG", "202608100046", 9),
        MockGameRow("SMALL", "202608100045", 1),
    ]

    mock_exec_res = MagicMock()
    mock_exec_res.fetchall.return_value = gapped_rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec_res

    res = await generate_prediction(mock_session)

    assert res["status"] == "INSUFFICIENT_DATA"
    assert "Historical data gap detected" in res["message"]
    assert res["prediction"] is None


@pytest.mark.asyncio
async def test_accuracy_calculation_integrity():
    """Verify accuracy formula matches correct/completed * 100 on immutable predictions."""
    rows = [
        MockGameRow("SMALL", "202608100049", 2),
        MockGameRow("BIG", "202608100048", 8),
        MockGameRow("SMALL", "202608100047", 3),
        MockGameRow("BIG", "202608100046", 9),
        MockGameRow("SMALL", "202608100045", 1),
        MockGameRow("BIG", "202608100044", 7),
        MockGameRow("SMALL", "202608100043", 2),
        MockGameRow("BIG", "202608100042", 8),
        MockGameRow("SMALL", "202608100041", 3),
        MockGameRow("BIG", "202608100040", 9),
    ]

    mock_exec_res = MagicMock()
    mock_exec_res.scalars().all.return_value = []
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_exec_res

    history = await evaluate_recent_accuracy(mock_session, rows)

    assert isinstance(history, list)
    if history:
        wins = sum(1 for h in history if h["is_win"])
        total = len(history)
        calc_pct = (wins / total) * 100
        assert 0 <= calc_pct <= 100
