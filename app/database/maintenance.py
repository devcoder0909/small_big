"""Database maintenance — cleanup, retention, and health checks."""

import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_settings
from app.core.database import async_session_factory
from app.core.logging import setup_logging, get_logger
from app.models.raw_response import RawResponse
from app.models.game_result import GameResult

logger = get_logger(__name__)


async def cleanup_old_raw_responses():
    """
    Remove raw responses older than retention period.

    NEVER deletes game_results — only raw_responses.
    """
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.raw_response_retention_days)

    async with async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                delete(RawResponse).where(RawResponse.created_at < cutoff)
            )
            deleted = result.rowcount
            logger.info("raw_response_cleanup", deleted=deleted, cutoff=cutoff.isoformat())

    return {"deleted": deleted, "cutoff": cutoff.isoformat()}


async def prune_oldest_game_results(max_records: int | None = None):
    """
    Ensure total game_results count never exceeds max_records.

    Deletes records older than the max_records threshold.
    Prevents PostgreSQL disk space exhaustion while preserving large prediction window depth.
    """
    if max_records is None:
        settings = get_settings()
        max_records = settings.max_game_results_retention

    async with async_session_factory() as session:
        async with session.begin():
            # Find the threshold issue_id for the max_records-th record
            query = (
                select(GameResult.issue_id)
                .order_by(GameResult.issue_id.desc())
                .offset(max_records)
                .limit(1)
            )
            result = await session.execute(query)
            cutoff_issue = result.scalar_one_or_none()

            if cutoff_issue:
                del_result = await session.execute(
                    delete(GameResult).where(GameResult.issue_id < cutoff_issue)
                )
                pruned_count = del_result.rowcount
                logger.info(
                    "game_results_pruned",
                    pruned=pruned_count,
                    cutoff_issue=cutoff_issue,
                    max_records=max_records,
                )
                return {"pruned": pruned_count, "cutoff_issue": cutoff_issue}

    return {"pruned": 0, "cutoff_issue": None}


async def run_maintenance():
    """Run all maintenance tasks."""
    setup_logging()
    logger.info("maintenance_starting")

    results = {}

    try:
        results["raw_cleanup"] = await cleanup_old_raw_responses()
    except Exception as e:
        logger.error("maintenance_error", task="raw_cleanup", error=str(e))
        results["raw_cleanup"] = {"error": str(e)}

    try:
        settings = get_settings()
        results["game_prune"] = await prune_oldest_game_results(settings.max_game_results_retention)
    except Exception as e:
        logger.error("maintenance_error", task="game_prune", error=str(e))
        results["game_prune"] = {"error": str(e)}

    logger.info("maintenance_complete", results=results)
    return results


if __name__ == "__main__":
    asyncio.run(run_maintenance())
