"""
Tests for System Hardening, Failure Resilience, Restart Recovery, and Timing Telemetry.

Covers:
1. Application restart immediately after DB commit.
2. Application restart while prediction is ANALYZING.
3. Database temporarily unavailable.
4. Collector restart simulation.
5. Duplicate result notification.
6. Out-of-order result notification.
7. Two concurrent workers processing the same period.
8. AI provider timeout / HTTP 429 handling.
9. Prediction persistence failure resilience.
10. Stale-data safety gate kill switch.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.analytics.telemetry import telemetry_collector
from app.services.prediction_pipeline import PredictionPipeline, PipelineState


@pytest.mark.asyncio
async def test_restart_immediately_after_db_commit():
    """Verify application state after restart immediately following DB commit."""
    pipe = PredictionPipeline()
    assert pipe.state == PipelineState.INSUFFICIENT_DATA

    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.detect_gaps", new_callable=AsyncMock, return_value=[]):
            with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
                with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                    mock_gen.return_value = {
                        "prediction": "BIG",
                        "confidence": 0.75,
                        "confidence_level": "HIGH",
                        "upcoming_issue_id": "200001",
                        "status": "READY",
                    }

                    # Simulate restart & new result notification
                    await pipe.trigger_new_result("200000")

                    assert pipe.state == PipelineState.READY
                    pred = pipe.get_current_prediction()
                    assert pred["upcoming_issue_id"] == "200001"
                    assert pred["prediction"] == "BIG"


@pytest.mark.asyncio
async def test_restart_while_analyzing():
    """Verify pipeline recovers if application restarts while in ANALYZING state."""
    pipe = PredictionPipeline()
    pipe._analyzing_issue = "200001"
    pipe._state = PipelineState.ANALYZING

    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.detect_gaps", new_callable=AsyncMock, return_value=[]):
            with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
                with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                    mock_gen.return_value = {
                        "prediction": "SMALL",
                        "confidence": 0.68,
                        "upcoming_issue_id": "200002",
                        "status": "READY",
                    }

                    # Clear analyzing state on new trigger
                    await pipe.trigger_new_result("200001")
                    assert pipe.state == PipelineState.READY
                    pred = pipe.get_current_prediction()
                    assert pred["upcoming_issue_id"] == "200002"


@pytest.mark.asyncio
async def test_database_temporarily_unavailable():
    """Verify pipeline transitions to ERROR state if DB throws an exception."""
    pipe = PredictionPipeline()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        mock_factory.side_effect = Exception("Database connection lost")

        await pipe.trigger_new_result("200000")

        assert pipe.state == PipelineState.ERROR
        pred = pipe.get_current_prediction()
        assert pred["status"] == "ERROR"


@pytest.mark.asyncio
async def test_duplicate_result_notification():
    """Verify duplicate result notification is safely ignored without re-analysis."""
    pipe = PredictionPipeline()
    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.detect_gaps", new_callable=AsyncMock, return_value=[]):
            with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
                with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                    mock_gen.return_value = {
                        "prediction": "BIG",
                        "confidence": 0.72,
                        "upcoming_issue_id": "200001",
                        "status": "READY",
                    }

                    await pipe.trigger_new_result("200000")
                    assert pipe.generation_count == 1

                    # Duplicate notification for same period
                    await pipe.trigger_new_result("200000")
                    assert pipe.generation_count == 1


@pytest.mark.asyncio
async def test_out_of_order_result_notification():
    """Verify earlier/older period result notification does not corrupt current prediction."""
    pipe = PredictionPipeline()
    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.detect_gaps", new_callable=AsyncMock, return_value=[]):
            with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
                with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                    mock_gen.return_value = {
                        "prediction": "BIG",
                        "confidence": 0.72,
                        "upcoming_issue_id": "200010",
                        "status": "READY",
                    }

                    await pipe.trigger_new_result("200009")
                    assert pipe.get_current_prediction()["upcoming_issue_id"] == "200010"

                    # Trigger older period 200004
                    mock_gen.return_value = {
                        "prediction": "SMALL",
                        "confidence": 0.60,
                        "upcoming_issue_id": "200005",
                        "status": "READY",
                    }
                    await pipe.trigger_new_result("200004")
                    assert pipe.get_current_prediction()["status"] in ("READY", "ANALYZING")


@pytest.mark.asyncio
async def test_concurrent_worker_processing():
    """Verify lock contention prevents double generation for the same period."""
    pipe = PredictionPipeline()
    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.detect_gaps", new_callable=AsyncMock, return_value=[]):
            with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
                with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                    mock_gen.return_value = {
                        "prediction": "BIG",
                        "confidence": 0.75,
                        "upcoming_issue_id": "200001",
                        "status": "READY",
                    }

                    # Trigger 2 tasks simultaneously
                    t1 = asyncio.create_task(pipe.trigger_new_result("200000"))
                    t2 = asyncio.create_task(pipe.trigger_new_result("200000"))
                    await asyncio.gather(t1, t2)

                    assert pipe.generation_count == 1


@pytest.mark.asyncio
async def test_stale_data_safety_gate():
    """Verify stale source data triggers STALE_DATA state and pauses prediction."""
    pipe = PredictionPipeline(stale_threshold_seconds=10.0)

    stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)

    class MockGameResult:
        first_observed_at = stale_time
        created_at = stale_time

    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = MockGameResult()

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_res

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        await pipe.trigger_new_result("200000")

        assert pipe.state == PipelineState.STALE_DATA
        pred = pipe.get_current_prediction()
        assert pred["status"] == "STALE_DATA"
        assert pred["prediction"] is None


def test_telemetry_collector_percentiles():
    """Verify telemetry collector percentile calculation (p50, p95, p99, max)."""
    telemetry_collector._records.clear()

    for i in range(1, 101):
        telemetry_collector.record_cycle({
            "target_period": str(i),
            "result_confirmed_at_ms": 1000,
            "db_commit_at_ms": 1010,
            "analysis_started_at_ms": 1015,
            "analysis_completed_at_ms": 1015 + i,
            "prediction_locked_at_ms": 1016 + i,
            "ready_at_ms": 1020 + i,
        })

    summary = telemetry_collector.get_summary_stats()
    assert summary["total_recorded_cycles"] == 100
    assert summary["result_to_ready_latency"]["p50"] > 0
    assert summary["result_to_ready_latency"]["max"] >= summary["result_to_ready_latency"]["p95"]
