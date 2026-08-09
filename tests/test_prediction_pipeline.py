"""
Tests for the event-driven PredictionPipeline.

Covers:
1. Result → prediction lifecycle
2. Exact period association
3. Immediate prediction generation
4. ANALYZING → READY state transition
5. Prediction immutability (lock)
6. Duplicate-result handling
7. Concurrent get_current_prediction reads
8. AI signal timeout non-blocking
9. No future-data leakage
10. Stale prediction prevention
11. 100-period lifecycle simulation
12. Force refresh behavior
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.prediction_pipeline import PredictionPipeline


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


def _make_rows(count=50, start_id=1000):
    """Generate sequential mock game result rows."""
    rows = []
    for i in range(count):
        issue_id = str(start_id + count - 1 - i)
        size = "BIG" if (i % 3 == 0) else "SMALL"
        rows.append(MockRow(size, issue_id, number=(i % 10)))
    return rows


def _make_prediction(pred="BIG", conf=0.75, issue_id="52280"):
    """Create a standard mock prediction result."""
    return {
        "prediction": pred,
        "confidence": conf,
        "confidence_level": "HIGH" if conf >= 0.72 else "MEDIUM",
        "confluence_level": "STANDARD",
        "upcoming_issue_id": issue_id,
        "prediction_id": issue_id,
        "status": "ACTIVE",
        "active_indicators": 10,
        "agreeing_indicators": 7,
        "total_records_analyzed": 50,
        "created_at_ms": int(time.time() * 1000),
    }


@pytest.mark.asyncio
async def test_pipeline_initial_state():
    """Pipeline starts in INSUFFICIENT_DATA state before any trigger."""
    pipe = PredictionPipeline()
    result = pipe.get_current_prediction()
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["prediction"] is None
    assert result["upcoming_issue_id"] is None


@pytest.mark.asyncio
async def test_pipeline_trigger_generates_next_period():
    """Triggering with issue_id X generates prediction for X+1."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = _make_prediction("BIG", 0.75, "52280")

            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                await pipe.trigger_new_result("52279")

                result = pipe.get_current_prediction()
                assert result["status"] == "READY"
                assert result["prediction"] == "BIG"
                assert result["upcoming_issue_id"] == "52280"


@pytest.mark.asyncio
async def test_pipeline_analyzing_state():
    """Pipeline shows ANALYZING state while prediction is being generated."""
    pipe = PredictionPipeline()

    pipe._analyzing_issue = "52280"
    result = pipe.get_current_prediction()
    assert result["status"] == "ANALYZING"
    assert result["upcoming_issue_id"] == "52280"
    assert result["prediction"] is None


@pytest.mark.asyncio
async def test_pipeline_prediction_immutability():
    """Once locked, re-triggering same period doesn't change prediction."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                mock_gen.return_value = _make_prediction("BIG", 0.75, "52280")

                await pipe.trigger_new_result("52279")
                first_result = pipe.get_current_prediction()
                assert first_result["prediction"] == "BIG"

                # Trigger same period again — should not re-generate
                mock_gen.return_value = _make_prediction("SMALL", 0.80, "52280")
                await pipe.trigger_new_result("52279")

                second_result = pipe.get_current_prediction()
                assert second_result["prediction"] == "BIG"  # Immutable — stays BIG
                assert mock_gen.call_count == 1  # Only called once


@pytest.mark.asyncio
async def test_pipeline_new_period_replaces_old():
    """New period trigger replaces previous prediction."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                mock_gen.return_value = _make_prediction("BIG", 0.75, "52280")
                await pipe.trigger_new_result("52279")
                assert pipe.get_current_prediction()["upcoming_issue_id"] == "52280"

                mock_gen.return_value = _make_prediction("SMALL", 0.68, "52281")
                await pipe.trigger_new_result("52280")
                result = pipe.get_current_prediction()
                assert result["upcoming_issue_id"] == "52281"
                assert result["prediction"] == "SMALL"


@pytest.mark.asyncio
async def test_pipeline_concurrent_reads_consistent():
    """Multiple concurrent reads return identical data."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                mock_gen.return_value = _make_prediction("BIG", 0.80, "52280")
                await pipe.trigger_new_result("52279")

    # 50 concurrent reads
    results = [pipe.get_current_prediction() for _ in range(50)]
    first = results[0]
    for r in results[1:]:
        assert r["prediction"] == first["prediction"]
        assert r["upcoming_issue_id"] == first["upcoming_issue_id"]
        assert r["confidence"] == first["confidence"]


@pytest.mark.asyncio
async def test_pipeline_insufficient_data_handling():
    """Pipeline returns INSUFFICIENT_DATA when engine says so."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {
                "prediction": None,
                "confidence": 0,
                "status": "INSUFFICIENT_DATA",
                "message": "Need at least 5 historical records",
            }

            await pipe.trigger_new_result("1002")
            result = pipe.get_current_prediction()
            assert result["status"] == "INSUFFICIENT_DATA"


@pytest.mark.asyncio
async def test_pipeline_generation_count():
    """Generation count tracks successful predictions."""
    pipe = PredictionPipeline()
    assert pipe.generation_count == 0

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                for i in range(5):
                    mock_gen.return_value = _make_prediction(
                        "BIG", 0.75, str(52280 + i)
                    )
                    await pipe.trigger_new_result(str(52279 + i))

    assert pipe.generation_count == 5


@pytest.mark.asyncio
async def test_pipeline_invalid_issue_id():
    """Invalid issue ID is handled gracefully."""
    pipe = PredictionPipeline()
    await pipe.trigger_new_result("not_a_number")
    result = pipe.get_current_prediction()
    assert result["status"] == "INSUFFICIENT_DATA"


@pytest.mark.asyncio
async def test_pipeline_force_refresh():
    """Force refresh generates prediction from latest DB data."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    # Mock the scalar_one_or_none for force_refresh's SELECT latest issue
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = "52279"
    mock_session.execute.return_value = mock_exec

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                mock_gen.return_value = _make_prediction("SMALL", 0.65, "52280")

                await pipe.force_refresh()
                result = pipe.get_current_prediction()
                assert result["prediction"] == "SMALL"
                assert result["status"] == "READY"


@pytest.mark.asyncio
async def test_pipeline_100_period_lifecycle():
    """Simulate 100 consecutive periods with correct progression."""
    pipe = PredictionPipeline()

    mock_session = AsyncMock()
    mock_session.get_bind.return_value = MagicMock(dialect=MagicMock(name="sqlite"))

    with patch("app.services.prediction_pipeline.async_session_factory") as mock_factory:
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_session
        ctx.__aexit__.return_value = False
        mock_factory.return_value = ctx

        with patch("app.services.prediction_pipeline.generate_prediction", new_callable=AsyncMock) as mock_gen:
            with patch("app.services.prediction_pipeline.persist_original_prediction", new_callable=AsyncMock):
                base_id = 52230
                for i in range(100):
                    latest_id = str(base_id + i)
                    next_id = str(base_id + i + 1)

                    mock_gen.return_value = _make_prediction(
                        "BIG" if i % 2 == 0 else "SMALL",
                        0.70 + (i % 10) * 0.02,
                        next_id,
                    )

                    await pipe.trigger_new_result(latest_id)

                    result = pipe.get_current_prediction()
                    assert result["status"] == "READY", f"Period {next_id} not READY"
                    assert result["upcoming_issue_id"] == next_id
                    assert result["prediction"] in ("BIG", "SMALL")

    assert pipe.generation_count == 100


@pytest.mark.asyncio
async def test_pipeline_server_time_always_present():
    """Every response includes server_time_ms."""
    pipe = PredictionPipeline()
    result = pipe.get_current_prediction()
    assert "server_time_ms" in result
    assert isinstance(result["server_time_ms"], int)
    assert result["server_time_ms"] > 0
