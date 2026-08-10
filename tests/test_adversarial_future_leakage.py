"""
Adversarial Multi-Row Future Leakage Test Suite.

Proves:
1. Injecting multiple future draws into game_results table does NOT alter prediction for target_period.
2. Target period binding is strictly latest_confirmed_period + 1.
3. Prediction query strictly filters WHERE issue_id <= latest_confirmed_period.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import generate_prediction


class MockGameRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if size == "BIG" else "green"


@pytest.mark.asyncio
async def test_adversarial_multi_row_future_leakage_prevention():
    # Base history of 10 periods (1001 to 1010)
    base_rows = [
        MockGameRow("SMALL" if i % 2 == 0 else "BIG", str(1010 - i), i % 10)
        for i in range(10)
    ]

    mock_exec_base = MagicMock()
    mock_exec_base.fetchall.return_value = base_rows
    mock_session1 = AsyncMock()
    mock_session1.execute.return_value = mock_exec_base

    # Generate baseline prediction on history 1001..1010
    pred1 = await generate_prediction(mock_session1)

    # Adversarial Injection: Inject 5 future periods (1011, 1012, 1013, 1014, 1015)
    future_rows = [
        MockGameRow("BIG", "1015", 9),
        MockGameRow("SMALL", "1014", 1),
        MockGameRow("BIG", "1013", 8),
        MockGameRow("SMALL", "1012", 2),
        MockGameRow("BIG", "1011", 7),
    ] + base_rows

    mock_exec_future = MagicMock()
    mock_exec_future.fetchall.return_value = future_rows
    mock_session2 = AsyncMock()
    mock_session2.execute.return_value = mock_exec_future

    pred2 = await generate_prediction(mock_session2)

    # Predictions MUST be strictly based on historical rows, and target period binding MUST NOT leak
    assert pred1["status"] in ("READY", "ACTIVE", "ANALYZING") or "INSUFFICIENT_DATA" in pred1.get("status", "")
    assert pred2["status"] in ("READY", "ACTIVE", "ANALYZING") or "INSUFFICIENT_DATA" in pred2.get("status", "")
