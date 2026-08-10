"""Rolling statistics — sliding window analysis over recent results."""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult

ROLLING_WINDOWS = [20, 50, 100, 500, 1000, 2000, 5000, 10000]


async def calculate_rolling_stats(
    session: AsyncSession, max_records: int = 10000
) -> list[dict]:
    """
    Calculate rolling window statistics.

    Computes Small/Big percentages and change rates over sliding windows.

    Args:
        session: Database session.
        max_records: Maximum records to load for calculation.

    Returns:
        List of rolling statistics for each window size.
    """
    query = (
        select(GameResult.calculated_size, GameResult.issue_id)
        .order_by(desc(GameResult.issue_id))
        .limit(max_records)
    )

    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return []

    sizes = [row.calculated_size for row in rows]
    results = []

    for window in ROLLING_WINDOWS:
        if len(sizes) < window:
            continue

        window_data = sizes[:window]
        small_count = sum(1 for s in window_data if s == "SMALL")
        big_count = window - small_count

        # Calculate change rate (number of transitions / window)
        changes = sum(
            1 for i in range(1, len(window_data))
            if window_data[i] != window_data[i - 1]
        )
        change_rate = round(changes / (window - 1) * 100, 2) if window > 1 else 0

        # Calculate streak within window
        current_streak = 1
        for i in range(1, len(window_data)):
            if window_data[i] == window_data[0]:
                current_streak += 1
            else:
                break

        results.append({
            "window": window,
            "available_records": len(sizes),
            "small_count": small_count,
            "big_count": big_count,
            "small_percentage": round(small_count / window * 100, 2),
            "big_percentage": round(big_count / window * 100, 2),
            "change_rate": change_rate,
            "current_streak": current_streak,
            "current_streak_size": window_data[0],
            "label": "HISTORICAL STATISTICS",
        })

    return results
