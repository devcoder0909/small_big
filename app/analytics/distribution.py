"""Distribution analysis — result value and color distributions."""

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


async def calculate_distribution(
    session: AsyncSession, window: int | None = None
) -> dict:
    """
    Calculate comprehensive distribution statistics.

    Args:
        session: Database session.
        window: Number of recent records. None = all-time.

    Returns:
        Dict with distribution data.
    """
    # Get recent results
    query = (
        select(GameResult.result_number, GameResult.source_color, GameResult.calculated_size)
        .order_by(desc(GameResult.issue_id))
    )
    if window:
        query = query.limit(window)

    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return {
            "window": window or "all",
            "total": 0,
            "number_distribution": {},
            "color_distribution": {},
            "size_distribution": {},
            "label": "HISTORICAL STATISTICS",
        }

    total = len(rows)

    # Number distribution (0-9)
    number_dist = {i: 0 for i in range(10)}
    color_dist = {}
    size_dist = {"SMALL": 0, "BIG": 0}

    for row in rows:
        number_dist[row.result_number] = number_dist.get(row.result_number, 0) + 1
        # Parse multi-value color field
        colors = [c.strip() for c in row.source_color.split(",")]
        for color in colors:
            color_dist[color] = color_dist.get(color, 0) + 1
        size_dist[row.calculated_size] = size_dist.get(row.calculated_size, 0) + 1

    return {
        "window": window or "all",
        "total": total,
        "number_distribution": {
            str(k): {
                "count": v,
                "percentage": round(v / total * 100, 2) if total > 0 else 0,
            }
            for k, v in sorted(number_dist.items())
        },
        "color_distribution": {
            k: {
                "count": v,
                "percentage": round(v / total * 100, 2) if total > 0 else 0,
            }
            for k, v in sorted(color_dist.items())
        },
        "size_distribution": {
            k: {
                "count": v,
                "percentage": round(v / total * 100, 2) if total > 0 else 0,
            }
            for k, v in size_dist.items()
        },
        "label": "HISTORICAL STATISTICS",
    }
