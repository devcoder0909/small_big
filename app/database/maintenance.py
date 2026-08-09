"""Database maintenance — cleanup, retention, and health checks."""

import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_settings
from app.core.database import async_session_factory
from app.core.logging import setup_logging, get_logger
from app.models.raw_response import RawResponse

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

    logger.info("maintenance_complete", results=results)
    return results


if __name__ == "__main__":
    asyncio.run(run_maintenance())
