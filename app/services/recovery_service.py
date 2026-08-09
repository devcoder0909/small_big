"""Recovery service — handles missing data detection and recovery after downtime."""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game_result import GameResult
from app.models.data_quality import DataQualityEvent
from app.collector.client import SourceClient
from app.collector.parser import parse_history_response, extract_service_time
from app.collector.validator import validate_batch
from app.collector.deduplicator import upsert_batch
from app.core import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def recover_missing_records(session: AsyncSession) -> dict:
    """
    Recover missing records after downtime.

    Process:
    1. Get latest known issue from database
    2. Fetch current source history
    3. Identify any records newer than our latest
    4. Insert missing records

    Returns:
        Dict with recovery results.
    """
    settings = get_settings()

    # Get our latest known issue
    result = await session.execute(
        select(GameResult.issue_id)
        .order_by(desc(GameResult.issue_id))
        .limit(1)
    )
    latest_known = result.scalar_one_or_none()

    # Fetch from source (deep page size 50 for max historical coverage)
    client = SourceClient()
    try:
        fetch_result = await client.fetch_history(page_no=1, page_size=50)
    finally:
        await client.close()

    if not fetch_result.success or not fetch_result.data:
        logger.error("recovery_fetch_failed", error=fetch_result.error_message)
        return {
            "status": "FETCH_FAILED",
            "error": fetch_result.error_message,
            "recovered": 0,
        }

    # Parse
    try:
        parsed = parse_history_response(fetch_result.data)
    except ValueError as e:
        return {"status": "PARSE_FAILED", "error": str(e), "recovered": 0}

    valid, errors = validate_batch(parsed)

    if not valid:
        return {"status": "NO_VALID_RECORDS", "recovered": 0}

    # Find records we don't have
    new_records = []
    if latest_known:
        for r in valid:
            if r.issue_id > latest_known:
                new_records.append(r)
    else:
        new_records = valid  # First run — insert all

    if not new_records:
        logger.info("recovery_no_missing_records")
        return {"status": "UP_TO_DATE", "recovered": 0}

    # Insert missing records
    source_time = extract_service_time(fetch_result.data)
    batch_result = await upsert_batch(
        session,
        new_records,
        source_url=settings.source_url,
        raw_response_id=None,
        source_created_at=source_time,
    )

    logger.info(
        "recovery_complete",
        recovered=batch_result["new_records"],
        latest_before=latest_known,
        latest_after=new_records[0].issue_id if new_records else None,
    )

    return {
        "status": "RECOVERED",
        "recovered": batch_result["new_records"],
        "duplicates_skipped": batch_result["duplicates"],
        "latest_before": latest_known,
        "latest_after": new_records[0].issue_id if new_records else None,
    }


async def detect_gaps(session: AsyncSession, window: int = 100) -> list[dict]:
    """
    Detect gaps in sequential issue IDs.

    Args:
        session: Database session.
        window: Number of recent records to check.

    Returns:
        List of detected gaps.
    """
    result = await session.execute(
        select(GameResult.issue_id)
        .order_by(desc(GameResult.issue_id))
        .limit(window)
    )
    rows = result.fetchall()

    if len(rows) < 2:
        return []

    ids = sorted([row.issue_id for row in rows])
    gaps = []

    for i in range(1, len(ids)):
        try:
            prev_num = int(ids[i - 1])
            curr_num = int(ids[i])
            if curr_num - prev_num > 1:
                missing = list(range(prev_num + 1, curr_num))
                gaps.append({
                    "between": (ids[i - 1], ids[i]),
                    "missing_count": len(missing),
                    "missing_ids": [str(m) for m in missing[:10]],  # Cap at 10
                })
        except ValueError:
            continue

    return gaps
