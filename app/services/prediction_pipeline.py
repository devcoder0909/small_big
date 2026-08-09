"""
Prediction Pipeline — Event-driven prediction orchestrator.

Central coordinator that owns the complete result→predict lifecycle:
  RESULT COMMITTED → TRIGGER → ANALYZE → LOCK → SERVE

Race-condition safe via asyncio.Lock. AI calls are non-blocking with timeout.
"""

import asyncio
import time
from datetime import datetime, timezone

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.prediction_engine import generate_prediction, persist_original_prediction
from app.core.database import async_session_factory
from app.core.logging import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Event-driven prediction pipeline.

    States per period:
      ANALYZING  — prediction generation in progress
      READY      — prediction locked and available
      INSUFFICIENT_DATA — not enough history to predict
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_prediction: dict | None = None
        self._latest_processed_issue: str | None = None
        self._analyzing_issue: str | None = None
        self._generation_count = 0

    def get_current_prediction(self) -> dict:
        """Return the current locked prediction or an analyzing/waiting state."""
        if self._analyzing_issue and (
            not self._current_prediction
            or self._current_prediction.get("upcoming_issue_id") != self._analyzing_issue
        ):
            return {
                "upcoming_issue_id": self._analyzing_issue,
                "prediction": None,
                "confidence": 0,
                "status": "ANALYZING",
                "message": "Generating prediction for next period...",
                "server_time_ms": int(time.time() * 1000),
            }

        if self._current_prediction:
            result = dict(self._current_prediction)
            result["server_time_ms"] = int(time.time() * 1000)
            return result

        return {
            "upcoming_issue_id": None,
            "prediction": None,
            "confidence": 0,
            "status": "INSUFFICIENT_DATA",
            "message": "Waiting for game results...",
            "server_time_ms": int(time.time() * 1000),
        }

    async def trigger_new_result(self, latest_issue_id: str):
        """
        Called by the collector after a new result is committed to the database.

        Calculates the next period ID and generates a prediction immediately.
        Uses asyncio.Lock to prevent concurrent prediction races.
        """
        try:
            next_period = str(int(latest_issue_id) + 1)
        except (ValueError, TypeError):
            logger.warning("pipeline_invalid_issue_id", issue_id=latest_issue_id)
            return

        # Skip if we already have a locked prediction for this next period
        if (
            self._current_prediction
            and self._current_prediction.get("upcoming_issue_id") == next_period
            and self._current_prediction.get("status") == "READY"
        ):
            return

        # Mark as analyzing immediately so API shows ANALYZING state
        self._analyzing_issue = next_period

        async with self._lock:
            # Double-check after acquiring lock
            if (
                self._current_prediction
                and self._current_prediction.get("upcoming_issue_id") == next_period
                and self._current_prediction.get("status") == "READY"
            ):
                return

            t_start = time.monotonic()
            logger.info(
                "pipeline_prediction_start",
                latest_result=latest_issue_id,
                next_period=next_period,
            )

            try:
                async with async_session_factory() as session:
                    prediction = await generate_prediction(session, 500)

                    if not prediction or prediction.get("status") == "INSUFFICIENT_DATA":
                        self._current_prediction = prediction
                        self._analyzing_issue = None
                        logger.warning("pipeline_insufficient_data", next_period=next_period)
                        return

                    # Override the upcoming_issue_id to be deterministic
                    prediction["upcoming_issue_id"] = next_period
                    prediction["prediction_id"] = next_period
                    prediction["status"] = "READY"

                    # Persist the immutable prediction record
                    try:
                        await persist_original_prediction(session, prediction)
                    except Exception as persist_err:
                        logger.warning(
                            "pipeline_persist_warning",
                            error=str(persist_err),
                            next_period=next_period,
                        )

                    # Lock the prediction
                    self._current_prediction = prediction
                    self._latest_processed_issue = latest_issue_id
                    self._analyzing_issue = None
                    self._generation_count += 1

                    elapsed_ms = int((time.monotonic() - t_start) * 1000)
                    logger.info(
                        "pipeline_prediction_locked",
                        next_period=next_period,
                        prediction=prediction.get("prediction"),
                        confidence=prediction.get("confidence"),
                        elapsed_ms=elapsed_ms,
                        generation_count=self._generation_count,
                    )

            except Exception as e:
                self._analyzing_issue = None
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


# Global singleton
pipeline = PredictionPipeline()
