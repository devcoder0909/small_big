"""Result service — CRUD operations for game results."""

from datetime import datetime, timezone
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


async def get_latest_result(session: AsyncSession) -> dict | None:
    """Get the most recent observed game result."""
    result = await session.execute(
        select(GameResult)
        .order_by(desc(GameResult.issue_id))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return None

    return {
        "issue_id": row.issue_id,
        "result": row.result_number,
        "size": row.calculated_size,
        "color": row.source_color,
        "observed_at": row.last_observed_at.isoformat() if row.last_observed_at else None,
        "source_verified": True,
        "label": "ACTUAL RESULT",
    }


async def get_results(
    session: AsyncSession, limit: int = 20, offset: int = 0
) -> dict:
    """Get paginated game results."""
    # Count total
    count_result = await session.execute(
        select(func.count()).select_from(GameResult)
    )
    total = count_result.scalar() or 0

    # Fetch page
    result = await session.execute(
        select(GameResult)
        .order_by(desc(GameResult.issue_id))
        .limit(limit)
        .offset(offset)
    )
    rows = result.scalars().all()

    return {
        "results": [
            {
                "issue_id": r.issue_id,
                "result": r.result_number,
                "size": r.calculated_size,
                "color": r.source_color,
                "observed_at": r.last_observed_at.isoformat() if r.last_observed_at else None,
            }
            for r in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
        "label": "ACTUAL RESULTS",
    }


async def get_result_by_issue(session: AsyncSession, issue_id: str) -> dict | None:
    """Get a specific result by issue ID."""
    result = await session.execute(
        select(GameResult).where(GameResult.issue_id == issue_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        return None

    return {
        "issue_id": row.issue_id,
        "result": row.result_number,
        "size": row.calculated_size,
        "color": row.source_color,
        "premium": row.premium,
        "sum_value": row.sum_value,
        "first_observed_at": row.first_observed_at.isoformat() if row.first_observed_at else None,
        "last_observed_at": row.last_observed_at.isoformat() if row.last_observed_at else None,
        "data_hash": row.data_hash,
        "label": "ACTUAL RESULT",
    }


async def get_total_count(session: AsyncSession) -> int:
    """Get total number of game results."""
    result = await session.execute(
        select(func.count()).select_from(GameResult)
    )
    return result.scalar() or 0
