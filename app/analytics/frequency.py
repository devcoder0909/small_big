"""Frequency analysis — historical Small/Big distribution across configurable windows."""

from sqlalchemy import select, func, case, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult

WINDOWS = [20, 50, 100, 500, 1000, 2000, 5000, 10000]


async def calculate_frequency(
    session: AsyncSession, window: int | None = None
) -> dict:
    """
    Calculate Small/Big frequency distribution.

    HISTORICAL STATISTICS — not prediction.

    Args:
        session: Database session.
        window: Number of recent records. None = all-time.

    Returns:
        Dict with frequency statistics.
    """
    query = select(
        func.count().label("total"),
        func.sum(case((GameResult.calculated_size == "SMALL", 1), else_=0)).label("small_count"),
        func.sum(case((GameResult.calculated_size == "BIG", 1), else_=0)).label("big_count"),
    )

    if window:
        # Get the IDs of the most recent N records
        subq = (
            select(GameResult.id)
            .order_by(desc(GameResult.issue_id))
            .limit(window)
        ).subquery()
        query = query.where(GameResult.id.in_(select(subq.c.id)))

    result = await session.execute(query)
    row = result.fetchone()

    total = row.total or 0
    small = row.small_count or 0
    big = row.big_count or 0

    return {
        "window": window or "all",
        "total": total,
        "small_count": small,
        "big_count": big,
        "small_percentage": round((small / total * 100), 2) if total > 0 else 0,
        "big_percentage": round((big / total * 100), 2) if total > 0 else 0,
        "label": "HISTORICAL STATISTICS",
    }


async def calculate_all_frequencies(session: AsyncSession) -> list[dict]:
    """Calculate frequency for all standard windows + all-time."""
    results = []
    for w in WINDOWS:
        results.append(await calculate_frequency(session, w))
    results.append(await calculate_frequency(session, None))  # all-time
    return results


async def calculate_number_distribution(
    session: AsyncSession, window: int | None = None
) -> dict:
    """Calculate distribution of individual result numbers (0-9)."""
    if window:
        subq = (
            select(GameResult.id)
            .order_by(desc(GameResult.issue_id))
            .limit(window)
        ).subquery()
        query = (
            select(
                GameResult.result_number,
                func.count().label("count"),
            )
            .where(GameResult.id.in_(select(subq.c.id)))
            .group_by(GameResult.result_number)
            .order_by(GameResult.result_number)
        )
    else:
        query = (
            select(
                GameResult.result_number,
                func.count().label("count"),
            )
            .group_by(GameResult.result_number)
            .order_by(GameResult.result_number)
        )

    result = await session.execute(query)
    rows = result.fetchall()

    distribution = {i: 0 for i in range(10)}
    total = 0
    for row in rows:
        distribution[row.result_number] = row.count
        total += row.count

    return {
        "window": window or "all",
        "total": total,
        "distribution": distribution,
        "percentages": {
            k: round(v / total * 100, 2) if total > 0 else 0
            for k, v in distribution.items()
        },
        "label": "HISTORICAL STATISTICS",
    }
