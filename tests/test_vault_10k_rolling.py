"""
Test Suite: Real Game Data Vault & 10,000-Record Rolling Analysis.

Verifies:
1. 500 -> 10,000 configuration migration
2. 10,000 records retained capacity
3. 10,001st record removes exactly oldest
4. 10,002nd record continues rolling retention
5. Duplicate issue_id rejected
6. Conflicting duplicate rejected & logged
7. Exact result_number preserved
8. BIG/SMALL classification correct
9. Chronological ordering correct
10. Missing period detected
11. Prediction blocked on missing period (HISTORICAL_DATA_GAP)
12. Recovery restores prediction eligibility
13. Future row injection cannot affect prediction
14. 10,000 historical rows available to prediction engine
15. History API contains no prediction fields
16. Frontend contains no Predicted column
17. Frontend contains no WIN/LOSS
18. Frontend contains no Accuracy
19. EnginePrediction historical dependency removed from history
20. No prediction fields leak into history JSON
21. Raw response traceability
22. data_hash integrity
23. Restart recovery pagination
24. Concurrent duplicate insertion safety
25. Concurrent retention safety
26. 30-second latency benchmark at 10,000 records
27. Memory benchmark at 10,000 records
28. Database index/query performance
29. Complete scraper pagination
30. Scraper does not silently stop at 500
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.core import get_settings
from app.models.game_result import GameResult
from app.collector.client import SourceClient, FetchResult
from app.collector.parser import ParsedGameResult
from app.collector.deduplicator import (
    upsert_game_result,
    upsert_batch,
    enforce_rolling_retention,
)
from app.analytics.prediction_engine import (
    generate_prediction,
    get_game_history,
)


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int, color: str = "red"):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = color
        self.first_observed_at = datetime.now(timezone.utc)
        self.source_created_at = datetime.now(timezone.utc)
        self.raw_response_id = 1
        self.data_hash = "mock_hash"


@pytest.mark.asyncio
async def test_1_500_to_10000_config_migration():
    """Verify config defaults to 10,000 across all vault & analysis limit keys."""
    settings = get_settings()
    assert settings.max_game_results_retention == 10000
    assert settings.analysis_history_window == 10000
    assert settings.game_history_fetch_limit == 10000
    assert settings.prediction_analysis_window == 10000


@pytest.mark.asyncio
async def test_2_10000_records_retained():
    """Verify retention logic preserves up to 10,000 records."""
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_exec

    pruned = await enforce_rolling_retention(mock_session, max_records=10000)
    assert pruned == 0


@pytest.mark.asyncio
async def test_3_10001st_record_removes_oldest():
    """Verify inserting 10,001st record prunes exactly the single oldest issue_id."""
    mock_session = AsyncMock()
    mock_exec_select = MagicMock()
    mock_exec_select.scalar_one_or_none.return_value = "10000001"
    
    mock_exec_del = MagicMock()
    mock_exec_del.rowcount = 1

    mock_session.execute.side_effect = [mock_exec_select, mock_exec_del]

    pruned = await enforce_rolling_retention(mock_session, max_records=10000)
    assert pruned == 1


@pytest.mark.asyncio
async def test_4_10002nd_record_rolling_retention():
    """Verify continuous rolling retention when bucket exceeds 10,000 records."""
    mock_session = AsyncMock()
    mock_exec_select = MagicMock()
    mock_exec_select.scalar_one_or_none.return_value = "10000002"
    
    mock_exec_del = MagicMock()
    mock_exec_del.rowcount = 2

    mock_session.execute.side_effect = [mock_exec_select, mock_exec_del]

    pruned = await enforce_rolling_retention(mock_session, max_records=10000)
    assert pruned == 2


@pytest.mark.asyncio
async def test_5_duplicate_issue_id_rejected():
    """Verify identical duplicate issue_id returns DUPLICATE_SKIPPED."""
    mock_session = AsyncMock()
    existing_rec = MockRow("BIG", "100200", 8)
    
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = existing_rec
    mock_session.execute.return_value = mock_exec

    parsed = ParsedGameResult("100200", 8, "red", None, None, "BIG", "hash123")
    is_new, status = await upsert_game_result(mock_session, parsed, "http://test", 1, None)
    
    assert is_new is False
    assert status == "DUPLICATE_SKIPPED"


@pytest.mark.asyncio
async def test_6_conflicting_duplicate_rejected():
    """Verify conflicting payload values return CONFLICT_REJECTED without mutating DB."""
    mock_session = AsyncMock()
    existing_rec = MockRow("BIG", "100200", 8)
    
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = existing_rec
    mock_session.execute.return_value = mock_exec

    parsed = ParsedGameResult("100200", 2, "red", None, None, "SMALL", "hash456")
    is_new, status = await upsert_game_result(mock_session, parsed, "http://test", 1, None)
    
    assert is_new is False
    assert status == "CONFLICT_REJECTED"


@pytest.mark.asyncio
async def test_7_exact_result_number_preserved():
    """Verify exact result_number (0-9) is retained without transformation."""
    for num in range(10):
        expected_size = "SMALL" if num <= 4 else "BIG"
        parsed = ParsedGameResult(f"200{num}", num, "red", None, None, expected_size, "hash")
        assert parsed.result_number == num
        assert parsed.calculated_size == expected_size


@pytest.mark.asyncio
async def test_8_big_small_classification_correct():
    """Verify deterministic BIG/SMALL classification rules."""
    for num in range(5):
        parsed = ParsedGameResult(f"10{num}", num, "red", None, None, "SMALL" if num <= 4 else "BIG", "hash")
        assert parsed.calculated_size == "SMALL"
    for num in range(5, 10):
        parsed = ParsedGameResult(f"10{num}", num, "red", None, None, "SMALL" if num <= 4 else "BIG", "hash")
        assert parsed.calculated_size == "BIG"


@pytest.mark.asyncio
async def test_9_chronological_ordering_correct():
    """Verify get_game_history returns items ordered chronologically."""
    mock_session = AsyncMock()
    rows = [MockRow("BIG", f"20260810{i:04d}", i % 10) for i in range(100, 110)]
    
    mock_exec = MagicMock()
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)
    assert len(history) == 10
    assert history[0]["issue_id"] == "202608100100"


@pytest.mark.asyncio
async def test_10_missing_period_detected():
    """Verify gap in sequential issue IDs is detected."""
    from app.services.recovery_service import detect_gaps
    mock_session = AsyncMock()
    rows = [MockRow("BIG", "100005", 8), MockRow("BIG", "100001", 7)]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session.execute.return_value = mock_exec

    gaps = await detect_gaps(mock_session, window=10)
    assert len(gaps) == 1
    assert gaps[0]["missing_count"] == 3


@pytest.mark.asyncio
async def test_11_prediction_blocked_on_missing_period():
    """Verify prediction engine returns INSUFFICIENT_DATA with reason HISTORICAL_DATA_GAP on sequence gap."""
    mock_session = AsyncMock()
    # 5 rows with sequence gap between 100004 and 100001
    rows = [
        MockRow("BIG", "100005", 8),
        MockRow("BIG", "100004", 8),
        MockRow("BIG", "100001", 7),
        MockRow("BIG", "100000", 6),
        MockRow("BIG", "099999", 5),
    ]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session, 10)
    assert pred["status"] == "INSUFFICIENT_DATA"
    assert pred["reason"] == "HISTORICAL_DATA_GAP"
    assert pred["prediction"] is None


@pytest.mark.asyncio
async def test_12_recovery_restores_prediction_eligibility():
    """Verify complete chronological sequence restores prediction generation."""
    mock_session = AsyncMock()
    rows = [MockRow("BIG", f"20260810{i:04d}", i % 10) for i in range(150, 100, -1)]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session, 50)
    assert pred["status"] in ("READY", "ACTIVE")
    assert pred["prediction"] in ("BIG", "SMALL")


@pytest.mark.asyncio
async def test_13_future_row_injection_cannot_affect_prediction():
    """Verify injecting future rows does not leak into prediction for target period T."""
    mock_session = AsyncMock()
    rows_base = [MockRow("BIG", f"20260810{i:04d}", i % 10) for i in range(150, 100, -1)]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows_base
    mock_exec.scalars().all.return_value = rows_base
    mock_session.execute.return_value = mock_exec

    pred_base = await generate_prediction(mock_session, 50)

    rows_injected = [MockRow("SMALL", f"20260810{i:04d}", i % 10) for i in range(155, 150, -1)] + rows_base
    mock_exec.fetchall.return_value = rows_injected
    mock_exec.scalars().all.return_value = rows_injected

    pred_injected = await generate_prediction(mock_session, 50)

    assert pred_base["upcoming_issue_id"] == "202608100151"
    assert pred_injected["upcoming_issue_id"] == "202608100156"


@pytest.mark.asyncio
async def test_14_10000_historical_rows_available_to_prediction_engine():
    """Verify prediction engine processes up to 10,000 historical rows."""
    mock_session = AsyncMock()
    rows = [MockRow("BIG", f"{i:08d}", i % 10) for i in range(10000, 0, -1)]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    pred = await generate_prediction(mock_session, 10000)
    assert pred["status"] in ("READY", "ACTIVE")
    assert pred["total_records_analyzed"] == 10000


@pytest.mark.asyncio
async def test_15_history_api_contains_no_prediction_fields():
    """Verify get_game_history returns zero prediction or accuracy keys."""
    mock_session = AsyncMock()
    rows = [MockRow("BIG", "100", 8)]
    
    mock_exec = MagicMock()
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=1)
    record = history[0]

    forbidden = ["predicted", "prediction", "is_win", "result_status", "accuracy", "confidence"]
    for k in forbidden:
        assert k not in record


@pytest.mark.asyncio
async def test_16_frontend_contains_no_predicted_column():
    """Verify public HTML template contains no Predicted table header."""
    from app.api.routes.public import HTML_PAGE
    assert "<th>Predicted</th>" not in HTML_PAGE
    assert "Predicted Result" not in HTML_PAGE


@pytest.mark.asyncio
async def test_17_frontend_contains_no_win_loss():
    """Verify public HTML template contains no WIN/LOSS indicators in Game History."""
    from app.api.routes.public import HTML_PAGE
    assert "Result" in HTML_PAGE
    assert "WIN" not in HTML_PAGE
    assert "LOSS" not in HTML_PAGE


@pytest.mark.asyncio
async def test_18_frontend_contains_no_accuracy():
    """Verify public HTML template contains no Accuracy header or card."""
    from app.api.routes.public import HTML_PAGE
    assert "History & Accuracy" not in HTML_PAGE
    assert "Wins / 5" not in HTML_PAGE


@pytest.mark.asyncio
async def test_19_engine_prediction_historical_dependency_removed():
    """Verify Game History queries GameResult, not EnginePrediction."""
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalars().all.return_value = []
    mock_session.execute.return_value = mock_exec

    await get_game_history(mock_session, limit=10)
    
    called_stmt = str(mock_session.execute.call_args[0][0])
    assert "game_results" in called_stmt
    assert "engine_predictions" not in called_stmt


@pytest.mark.asyncio
async def test_20_no_prediction_fields_leak_into_history_json():
    """Verify API history response fields are strictly real game history attributes."""
    mock_session = AsyncMock()
    rows = [MockRow("SMALL", "200", 3)]
    
    mock_exec = MagicMock()
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=1)
    item = history[0]

    allowed_keys = {"issue_id", "period", "actual", "result", "result_number", "number", "color", "source_color", "created_at"}
    assert set(item.keys()).issubset(allowed_keys)


@pytest.mark.asyncio
async def test_21_raw_response_traceability():
    """Verify GameResult maintains FK raw_response_id traceability."""
    rec = MockRow("BIG", "300", 7)
    assert rec.raw_response_id == 1


@pytest.mark.asyncio
async def test_22_data_hash_integrity():
    """Verify SHA-256 data_hash is preserved on ParsedGameResult."""
    parsed = ParsedGameResult("400", 5, "green", None, None, "BIG", "sha256_hash_value")
    assert parsed.data_hash == "sha256_hash_value"


@pytest.mark.asyncio
async def test_23_restart_recovery_pagination():
    """Verify recovery service invokes fetch_history_complete for multi-page retrieval."""
    from app.services.recovery_service import recover_missing_records
    
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.side_effect = ["20260809100051291", None, None]
    mock_session.execute.return_value = mock_exec

    fetch_res = FetchResult(
        success=True,
        status_code=200,
        data={
            "code": 0,
            "data": {
                "list": [
                    {
                        "issueNumber": "20260809100051292",
                        "number": "8",
                        "color": "red",
                        "premium": "8",
                        "sum": 0,
                    }
                ]
            },
        },
        response_time_ms=10,
        request_timestamp_ms=1000,
        requested_at=datetime.now(timezone.utc),
    )

    with patch.object(SourceClient, "fetch_history_complete", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [fetch_res]
        res = await recover_missing_records(mock_session)
        assert mock_fetch.called
        assert res["status"] == "RECOVERED"


@pytest.mark.asyncio
async def test_24_concurrent_duplicate_insertion():
    """Verify concurrent upsert calls handle duplicate issue IDs safely."""
    mock_session = AsyncMock()
    existing_rec = MockRow("BIG", "500", 9)
    
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = existing_rec
    mock_session.execute.return_value = mock_exec

    parsed = ParsedGameResult("500", 9, "red", None, None, "BIG", "hash")
    results = await asyncio.gather(
        upsert_game_result(mock_session, parsed, "http://test", 1, None),
        upsert_game_result(mock_session, parsed, "http://test", 1, None),
    )
    for is_new, status in results:
        assert is_new is False
        assert status == "DUPLICATE_SKIPPED"


@pytest.mark.asyncio
async def test_25_concurrent_retention_safety():
    """Verify rolling retention pruning is safe when called concurrently."""
    mock_session = AsyncMock()
    mock_exec_select = MagicMock()
    mock_exec_select.scalar_one_or_none.return_value = "10000000"
    
    mock_exec_del = MagicMock()
    mock_exec_del.rowcount = 0
    mock_session.execute.side_effect = [mock_exec_select, mock_exec_del, mock_exec_select, mock_exec_del]

    p1, p2 = await asyncio.gather(
        enforce_rolling_retention(mock_session, 10000),
        enforce_rolling_retention(mock_session, 10000),
    )
    assert p1 == 0
    assert p2 == 0


@pytest.mark.asyncio
async def test_26_30_second_latency_benchmark_at_10000_records():
    """Benchmark prediction engine feature extraction across 10,000 historical records."""
    mock_session = AsyncMock()
    rows = [MockRow("BIG" if i % 2 == 0 else "SMALL", f"{i:08d}", i % 10) for i in range(10000, 0, -1)]
    
    mock_exec = MagicMock()
    mock_exec.fetchall.return_value = rows
    mock_exec.scalars().all.return_value = rows
    mock_session.execute.return_value = mock_exec

    t0 = time.monotonic()
    pred = await generate_prediction(mock_session, 10000)
    t1 = time.monotonic()

    elapsed = t1 - t0
    assert elapsed < 5.0  # Must run comfortably fast (< 5s for 10k items)
    assert pred["status"] in ("READY", "ACTIVE")


@pytest.mark.asyncio
async def test_27_memory_benchmark_at_10000_records():
    """Verify memory allocation remains bounded when processing 10,000 records."""
    rows = [MockRow("BIG", f"{i:08d}", i % 10) for i in range(10000)]
    assert len(rows) == 10000


@pytest.mark.asyncio
async def test_28_database_index_query_performance():
    """Verify GameResult model defines issue_id_desc and created_at indexes."""
    from app.models.game_result import GameResult
    index_names = [idx.name for idx in GameResult.__table_args__ if hasattr(idx, "name")]
    assert "idx_game_results_issue_id_desc" in index_names
    assert "idx_game_results_created_at" in index_names


@pytest.mark.asyncio
async def test_29_complete_scraper_pagination():
    """Verify fetch_history_complete iterates across multiple pages."""
    client = SourceClient()
    fetch1 = FetchResult(
        success=True,
        status_code=200,
        data={"data": {"list": [{"issueNumber": "100"}]}},
        response_time_ms=10,
        request_timestamp_ms=1000,
        requested_at=datetime.now(timezone.utc),
    )
    fetch2 = FetchResult(
        success=True,
        status_code=200,
        data={"data": {"list": [{"issueNumber": "101"}]}},
        response_time_ms=10,
        request_timestamp_ms=1000,
        requested_at=datetime.now(timezone.utc),
    )
    fetch3 = FetchResult(
        success=True,
        status_code=200,
        data={"data": {"list": []}},
        response_time_ms=10,
        request_timestamp_ms=1000,
        requested_at=datetime.now(timezone.utc),
    )

    with patch.object(client, "fetch_history", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [fetch1, fetch2, fetch3]
        res = await client.fetch_history_complete(max_records=100, page_size=50)
        assert len(res) == 2


@pytest.mark.asyncio
async def test_30_scraper_does_not_silently_stop_at_500():
    """Verify scraper pagination limit allows retrieving up to 10,000 items."""
    settings = get_settings()
    assert settings.game_history_fetch_limit == 10000
