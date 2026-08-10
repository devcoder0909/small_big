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

    # Fetch complete history from source using pagination up to configured limit
    client = SourceClient()
    try:
        fetch_results = await client.fetch_history_complete(
            max_records=settings.game_history_fetch_limit, page_size=50
        )
    finally:
        await client.close()

    if not fetch_results:
        logger.error("recovery_fetch_failed", error="No response received from source API")
        return {
            "status": "FETCH_FAILED",
            "error": "No response received from source API",
            "recovered": 0,
        }

    # Aggregate & parse all fetched pages
    all_parsed = []
    source_time = None
    for res in fetch_results:
        if res.success and res.data:
            try:
                page_parsed = parse_history_response(res.data)
                all_parsed.extend(page_parsed)
                if not source_time:
                    source_time = extract_service_time(res.data)
            except ValueError:
                continue

    valid, errors = validate_batch(all_parsed)

    if not valid:
        return {"status": "NO_VALID_RECORDS", "recovered": 0}

    # Deduplicate & sort chronologically
    seen = set()
    deduped_valid = []
    for r in valid:
        if r.issue_id not in seen:
            seen.add(r.issue_id)
            deduped_valid.append(r)

    # Find records we don't have
    new_records = []
    if latest_known:
        for r in deduped_valid:
            if r.issue_id > latest_known:
                new_records.append(r)
    else:
        new_records = deduped_valid  # First run — insert all

    if not new_records:
        logger.info("recovery_no_missing_records")
        return {"status": "UP_TO_DATE", "recovered": 0}

    # Insert missing records
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
