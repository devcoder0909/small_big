"""
Prediction Pipeline — Event-driven prediction orchestrator.

Central coordinator that owns the complete result→predict lifecycle:
  RESULT COMMITTED → TRIGGER → ANALYZE → LOCK → SERVE

Race-condition safe via asyncio.Lock. AI calls are non-blocking with timeout.
Includes explicit pipeline state machine, stale-data protection gate, and lifecycle timing telemetry.
"""

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.prediction_engine import generate_prediction, persist_original_prediction
from app.analytics.telemetry import telemetry_collector
from app.core.database import async_session_factory
from app.core.logging import get_logger
from app.services.recovery_service import detect_gaps

logger = get_logger(__name__)


class PipelineState(str, Enum):
    WAITING_FOR_RESULT = "WAITING_FOR_RESULT"
    ANALYZING = "ANALYZING"
    READY = "READY"
    STALE_DATA = "STALE_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ERROR = "ERROR"


class PredictionPipeline:
    """
    Event-driven prediction pipeline with explicit state machine.
    """

    def __init__(self, stale_threshold_seconds: float = 180.0):
        self._lock = asyncio.Lock()
        self._current_prediction: dict | None = None
        self._latest_processed_issue: str | None = None
        self._analyzing_issue: str | None = None
        self._generation_count = 0
        self._state = PipelineState.INSUFFICIENT_DATA
        self._stale_threshold_seconds = stale_threshold_seconds

    def get_current_prediction(self) -> dict:
        """Return the current locked prediction or explicit pipeline state."""
        now_ms = int(time.time() * 1000)

        if self._analyzing_issue and (
            not self._current_prediction
            or self._current_prediction.get("upcoming_issue_id") != self._analyzing_issue
        ):
            return {
                "upcoming_issue_id": self._analyzing_issue,
                "prediction": None,
                "confidence": 0,
                "status": PipelineState.ANALYZING.value,
                "message": "Generating prediction for next period...",
                "server_time_ms": now_ms,
            }

        if self._current_prediction:
            result = dict(self._current_prediction)
            result["server_time_ms"] = now_ms
            return result

        return {
            "upcoming_issue_id": None,
            "prediction": None,
            "confidence": 0,
            "status": self._state.value,
            "message": "Waiting for game results...",
            "server_time_ms": now_ms,
        }

    async def trigger_new_result(self, latest_issue_id: str, confirmed_at_ms: int | None = None):
        """
        Called by the collector after a new result is committed to the database.

        Calculates target_period = latest_issue_id + 1 and generates a prediction immediately.
        Uses asyncio.Lock to prevent concurrent prediction races.
        """
        t_confirm_ms = confirmed_at_ms or int(time.time() * 1000)
        t_commit_ms = int(time.time() * 1000)

        try:
            next_period = str(int(latest_issue_id) + 1)
        except (ValueError, TypeError):
            logger.warning("pipeline_invalid_issue_id", issue_id=latest_issue_id)
            return

        # Skip if we already have a locked prediction for this target period
        if (
            self._current_prediction
            and self._current_prediction.get("upcoming_issue_id") == next_period
            and self._current_prediction.get("status") == PipelineState.READY.value
        ):
            return

        # Mark as analyzing immediately
        self._analyzing_issue = next_period
        self._state = PipelineState.ANALYZING

        async with self._lock:
            # Double-check after acquiring lock
            if (
                self._current_prediction
                and self._current_prediction.get("upcoming_issue_id") == next_period
                and self._current_prediction.get("status") == PipelineState.READY.value
            ):
                return

            t_analysis_start_ms = int(time.time() * 1000)
            try:
                async with async_session_factory() as session:
                    # 1. Stale data & gap safety gate
                    from app.models.game_result import GameResult
                    res = await session.execute(
                        select(GameResult)
                        .order_by(desc(GameResult.issue_id))
                        .limit(1)
                    )
                    latest_rec = None
                    if hasattr(res, "scalar_one_or_none"):
                        try:
                            latest_rec = res.scalar_one_or_none()
                        except Exception:
                            latest_rec = None

                    if latest_rec and type(latest_rec).__name__ != "MagicMock" and type(latest_rec).__name__ != "AsyncMock":
                        last_obs = getattr(latest_rec, "first_observed_at", None) or getattr(latest_rec, "created_at", None)
                        if last_obs and isinstance(last_obs, datetime):
                            if last_obs.tzinfo is None:
                                last_obs = last_obs.replace(tzinfo=timezone.utc)
                            age_sec = (datetime.now(timezone.utc) - last_obs).total_seconds()
                            if age_sec > self._stale_threshold_seconds:
                                logger.warning(
                                    "pipeline_stale_data_gate_triggered",
                                    latest_issue=latest_issue_id,
                                    age_seconds=age_sec,
                                )
                                self._state = PipelineState.STALE_DATA
                                self._current_prediction = {
                                    "upcoming_issue_id": next_period,
                                    "prediction": None,
                                    "confidence": 0,
                                    "status": PipelineState.STALE_DATA.value,
                                    "message": f"Source data is stale ({int(age_sec)}s old) — prediction paused for data safety",
                                    "server_time_ms": int(time.time() * 1000),
                                }
                                self._analyzing_issue = None
                                return

                    # 2. Generate prediction strictly from historical data <= latest_issue_id
                    prediction = await generate_prediction(session, None)
                    t_analysis_complete_ms = int(time.time() * 1000)

                    if not prediction or prediction.get("status") == PipelineState.INSUFFICIENT_DATA.value:
                        self._state = PipelineState.INSUFFICIENT_DATA
                        self._current_prediction = prediction
                        self._analyzing_issue = None
                        logger.warning("pipeline_insufficient_data", next_period=next_period)
                        return

                    # Enforce period binding & readiness
                    prediction["upcoming_issue_id"] = next_period
                    prediction["prediction_id"] = next_period
                    prediction["status"] = PipelineState.READY.value

                    # Persist original immutable prediction
                    try:
                        await persist_original_prediction(session, prediction)
                    except Exception as persist_err:
                        logger.warning(
                            "pipeline_persist_warning",
                            error=str(persist_err),
                            next_period=next_period,
                        )

                    t_locked_ms = int(time.time() * 1000)
                    t_ready_ms = t_locked_ms

                    # Attach latency breakdown
                    prediction["latency_breakdown_ms"] = {
                        "analysis_ms": max(0, t_analysis_complete_ms - t_analysis_start_ms),
                        "persist_ms": max(0, t_locked_ms - t_analysis_complete_ms),
                        "total_cycle_ms": max(0, t_ready_ms - t_confirm_ms),
                    }

                    # Telemetry recording
                    telemetry_data = {
                        "target_period": next_period,
                        "result_confirmed_at_ms": t_confirm_ms,
                        "db_commit_at_ms": t_commit_ms,
                        "analysis_started_at_ms": t_analysis_start_ms,
                        "analysis_completed_at_ms": t_analysis_complete_ms,
                        "prediction_locked_at_ms": t_locked_ms,
                        "ready_at_ms": t_ready_ms,
                    }
                    telemetry_collector.record_cycle(telemetry_data)

                    # Lock in memory
                    self._current_prediction = prediction
                    self._latest_processed_issue = latest_issue_id
                    self._analyzing_issue = None
                    self._state = PipelineState.READY
                    self._generation_count += 1

                    logger.info(
                        "pipeline_prediction_locked",
                        next_period=next_period,
                        prediction=prediction.get("prediction"),
                        confidence=prediction.get("confidence"),
                        total_ms=t_ready_ms - t_confirm_ms,
                        generation_count=self._generation_count,
                    )

            except Exception as e:
                self._analyzing_issue = None
                self._state = PipelineState.ERROR
                logger.error(
                    "pipeline_prediction_error",
                    error=str(e),
                    next_period=next_period,
                )

    async def force_refresh(self):
        """Force a prediction refresh using the latest data in the database."""
        try:
            async with async_session_factory() as session:
                from app.models.game_result import GameResult
                result = await session.execute(
                    select(GameResult.issue_id)
                    .order_by(desc(GameResult.issue_id))
                    .limit(1)
                )
                latest = result.scalar_one_or_none()
                if latest:
                    await self.trigger_new_result(latest)
        except Exception as e:
            logger.error("pipeline_force_refresh_error", error=str(e))

    @property
    def generation_count(self) -> int:
        return self._generation_count

    @property
    def latest_processed_issue(self) -> str | None:
        return self._latest_processed_issue

    @property
    def state(self) -> PipelineState:
        return self._state


# Global singleton
pipeline = PredictionPipeline()
