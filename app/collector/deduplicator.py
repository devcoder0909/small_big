"""Deduplicator module — prevents duplicate insertions using DB constraints."""

from datetime import datetime, timezone
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game_result import GameResult
from app.collector.parser import ParsedGameResult
from app.core import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def get_latest_issue_id(session: AsyncSession) -> str | None:
    """Get the latest (highest) issue_id from the database."""
    result = await session.execute(
        select(GameResult.issue_id)
        .order_by(GameResult.issue_id.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row


async def get_existing_issue_ids(
    session: AsyncSession, issue_ids: list[str]
) -> set[str]:
    """Check which issue_ids already exist in the database."""
    if not issue_ids:
        return set()

    result = await session.execute(
        select(GameResult.issue_id).where(GameResult.issue_id.in_(issue_ids))
    )
    return {row[0] for row in result.fetchall()}


async def enforce_rolling_retention(
    session: AsyncSession, max_records: int = 10000
) -> int:
    """
    Enforce exact rolling retention: delete records older than the newest max_records.
    Ensures COUNT(GameResult) <= max_records atomically in the active session.
    """
    query = (
        select(GameResult.issue_id)
        .order_by(GameResult.issue_id.desc())
        .offset(max_records)
        .limit(1)
    )
    result = await session.execute(query)
    cutoff_issue = result.scalar_one_or_none()

    if cutoff_issue:
        del_stmt = delete(GameResult).where(GameResult.issue_id <= cutoff_issue)
        del_result = await session.execute(del_stmt)
        pruned_count = del_result.rowcount
        logger.info(
            "rolling_retention_enforced",
            pruned=pruned_count,
            cutoff_issue=cutoff_issue,
            max_records=max_records,
        )
        return pruned_count
    return 0


async def upsert_game_result(
    session: AsyncSession,
    parsed: ParsedGameResult,
    source_url: str,
    raw_response_id: int | None,
    source_created_at: datetime | None,
) -> tuple[bool, str]:
    """
    Insert or validate a game result using idempotent insertion & conflict protection.

    Args:
        session: Active database session.
        parsed: Parsed game result.
        source_url: URL the data was fetched from.
        raw_response_id: FK to raw_responses table.
        source_created_at: Timestamp from the source API.

    Returns:
        Tuple of (is_new, status_message).
    """
    now = datetime.now(timezone.utc)
    existing_stmt = select(GameResult).where(GameResult.issue_id == parsed.issue_id)
    existing_res = await session.execute(existing_stmt)
    existing_rec = existing_res.scalar_one_or_none()

    if existing_rec is None:
        new_record = GameResult(
            issue_id=parsed.issue_id,
            result_number=parsed.result_number,
            source_color=parsed.source_color,
            premium=parsed.premium,
            sum_value=parsed.sum_value,
            calculated_size=parsed.calculated_size,
            source_created_at=source_created_at,
            first_observed_at=now,
            last_observed_at=now,
            source_url=source_url,
            raw_response_id=raw_response_id,
            data_hash=parsed.data_hash,
            created_at=now,
            updated_at=now,
        )
        session.add(new_record)
        logger.info("new_record", issue_id=parsed.issue_id, size=parsed.calculated_size)
        return True, "NEW_RECORD_DETECTED"
    else:
        # Check for conflicting source payload values (Case B)
        if (
            existing_rec.result_number != parsed.result_number
            or existing_rec.calculated_size != parsed.calculated_size
        ):
            logger.error(
                "data_integrity_conflict_detected",
                issue_id=parsed.issue_id,
                existing_number=existing_rec.result_number,
                existing_size=existing_rec.calculated_size,
                incoming_number=parsed.result_number,
                incoming_size=parsed.calculated_size,
            )
            return False, "CONFLICT_REJECTED"

        # Case A: Identical authoritative values -> safe duplicate skip
        logger.debug("duplicate_record", issue_id=parsed.issue_id)
        return False, "DUPLICATE_SKIPPED"


async def upsert_batch(
    session: AsyncSession,
    parsed_results: list[ParsedGameResult],
    source_url: str,
    raw_response_id: int | None,
    source_created_at: datetime | None,
) -> dict:
    """
    Upsert a batch of parsed results and enforce rolling retention.

    Returns:
        Dict with counts: new_records, duplicates, errors, pruned.
    """
    settings = get_settings()
    new_count = 0
    dup_count = 0
    error_count = 0

    for parsed in parsed_results:
        try:
            is_new, status = await upsert_game_result(
                session, parsed, source_url, raw_response_id, source_created_at
            )
            if is_new:
                new_count += 1
            else:
                dup_count += 1
        except Exception as e:
            logger.error(
                "upsert_error",
                issue_id=parsed.issue_id,
                error=str(e),
            )
            error_count += 1

    pruned = 0
    if new_count > 0:
        pruned = await enforce_rolling_retention(
            session, max_records=settings.max_game_results_retention
        )

    return {
        "new_records": new_count,
        "duplicates": dup_count,
        "errors": error_count,
        "pruned": pruned,
    }


async def get_total_record_count(session: AsyncSession) -> int:
    """Get total count of game results in the database."""
    result = await session.execute(
        select(func.count()).select_from(GameResult)
    )
    return result.scalar() or 0
