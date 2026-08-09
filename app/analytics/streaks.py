"""Streak analysis — historical streak detection and statistics."""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


async def calculate_streaks(
    session: AsyncSession, window: int | None = None
) -> dict:
    """
    Calculate historical streak statistics.

    HISTORICAL STATISTICS — a streak ending does NOT guarantee reversal.

    Args:
        session: Database session.
        window: Number of recent records. None = all-time.

    Returns:
        Dict with streak statistics.
    """
    query = (
        select(GameResult.calculated_size, GameResult.issue_id)
        .order_by(desc(GameResult.issue_id))
    )
    if window:
        query = query.limit(window)

    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return {
            "current_size": None,
            "current_streak": 0,
            "longest_small_streak": 0,
            "longest_big_streak": 0,
            "average_small_streak": 0.0,
            "average_big_streak": 0.0,
            "total_streaks": 0,
            "streak_distribution": {},
            "label": "HISTORICAL STATISTICS",
        }

    sizes = [row.calculated_size for row in rows]

    # Current streak (from most recent)
    current_size = sizes[0]
    current_streak = 1
    for i in range(1, len(sizes)):
        if sizes[i] == current_size:
            current_streak += 1
        else:
            break

    # All streaks
    streaks = {"SMALL": [], "BIG": []}
    streak_size = sizes[0]
    streak_length = 1

    for i in range(1, len(sizes)):
        if sizes[i] == streak_size:
            streak_length += 1
        else:
            streaks[streak_size].append(streak_length)
            streak_size = sizes[i]
            streak_length = 1
    streaks[streak_size].append(streak_length)

    longest_small = max(streaks["SMALL"]) if streaks["SMALL"] else 0
    longest_big = max(streaks["BIG"]) if streaks["BIG"] else 0

    avg_small = (
        round(sum(streaks["SMALL"]) / len(streaks["SMALL"]), 2)
        if streaks["SMALL"]
        else 0.0
    )
    avg_big = (
        round(sum(streaks["BIG"]) / len(streaks["BIG"]), 2)
        if streaks["BIG"]
        else 0.0
    )

    # Streak length distribution
    all_lengths = streaks["SMALL"] + streaks["BIG"]
    distribution = {}
    for length in all_lengths:
        distribution[length] = distribution.get(length, 0) + 1

    return {
        "current_size": current_size,
        "current_streak": current_streak,
        "longest_small_streak": longest_small,
        "longest_big_streak": longest_big,
        "average_small_streak": avg_small,
        "average_big_streak": avg_big,
        "total_streaks": len(all_lengths),
        "small_streaks_count": len(streaks["SMALL"]),
        "big_streaks_count": len(streaks["BIG"]),
        "streak_distribution": dict(sorted(distribution.items())),
        "label": "HISTORICAL STATISTICS",
    }
