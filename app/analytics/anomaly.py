"""Anomaly detection — statistical anomaly indicators."""

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


async def detect_anomalies(
    session: AsyncSession, window: int = 200
) -> dict:
    """
    Detect statistical anomalies in recent results.

    Anomaly types:
    - Long streaks (>= 2x average streak length)
    - Frequency deviation (>= 10% from 50/50)
    - Distribution anomaly (any number appearing > 2x expected)

    Output levels:
    - NORMAL: No significant anomalies
    - WATCH: Minor deviation detected
    - ANOMALY: Significant statistical deviation

    Args:
        session: Database session.
        window: Number of recent records to analyze.

    Returns:
        Dict with anomaly indicators.
    """
    query = (
        select(GameResult.calculated_size, GameResult.result_number)
        .order_by(desc(GameResult.issue_id))
        .limit(window)
    )

    result = await session.execute(query)
    rows = result.fetchall()

    if len(rows) < 10:
        return {
            "status": "INSUFFICIENT_DATA",
            "indicators": [],
            "total_analyzed": len(rows),
            "label": "HISTORICAL STATISTICS",
        }

    sizes = [row.calculated_size for row in rows]
    numbers = [row.result_number for row in rows]
    indicators = []
    overall_status = "NORMAL"

    # 1. Current streak check
    current_streak = 1
    current_size = sizes[0]
    for i in range(1, len(sizes)):
        if sizes[i] == current_size:
            current_streak += 1
        else:
            break

    # Calculate average streak length
    streaks = []
    s_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == sizes[i - 1]:
            s_len += 1
        else:
            streaks.append(s_len)
            s_len = 1
    streaks.append(s_len)
    avg_streak = sum(streaks) / len(streaks) if streaks else 1

    if current_streak >= avg_streak * 3:
        indicators.append({
            "type": "long_streak",
            "level": "ANOMALY",
            "description": f"Current {current_size} streak of {current_streak} is {current_streak/avg_streak:.1f}x average ({avg_streak:.1f})",
            "value": current_streak,
            "threshold": round(avg_streak * 3, 1),
        })
        overall_status = "ANOMALY"
    elif current_streak >= avg_streak * 2:
        indicators.append({
            "type": "long_streak",
            "level": "WATCH",
            "description": f"Current {current_size} streak of {current_streak} is {current_streak/avg_streak:.1f}x average",
            "value": current_streak,
            "threshold": round(avg_streak * 2, 1),
        })
        if overall_status == "NORMAL":
            overall_status = "WATCH"

    # 2. Frequency deviation check
    small_count = sum(1 for s in sizes if s == "SMALL")
    small_pct = small_count / len(sizes) * 100
    deviation = abs(small_pct - 50)

    if deviation >= 15:
        indicators.append({
            "type": "frequency_deviation",
            "level": "ANOMALY",
            "description": f"Small/Big ratio {small_pct:.1f}%/{100-small_pct:.1f}% deviates {deviation:.1f}% from 50/50",
            "value": round(deviation, 2),
            "threshold": 15,
        })
        overall_status = "ANOMALY"
    elif deviation >= 8:
        indicators.append({
            "type": "frequency_deviation",
            "level": "WATCH",
            "description": f"Small/Big ratio {small_pct:.1f}%/{100-small_pct:.1f}% deviates {deviation:.1f}% from 50/50",
            "value": round(deviation, 2),
            "threshold": 8,
        })
        if overall_status == "NORMAL":
            overall_status = "WATCH"

    # 3. Number distribution anomaly
    expected_per_number = len(numbers) / 10
    number_counts = {i: 0 for i in range(10)}
    for n in numbers:
        number_counts[n] = number_counts.get(n, 0) + 1

    for num, count in number_counts.items():
        if expected_per_number > 0 and count > expected_per_number * 2.5:
            indicators.append({
                "type": "number_distribution_anomaly",
                "level": "WATCH",
                "description": f"Number {num} appeared {count} times (expected ~{expected_per_number:.0f})",
                "value": count,
                "threshold": round(expected_per_number * 2.5),
            })
            if overall_status == "NORMAL":
                overall_status = "WATCH"

    # 4. Consecutive same-number check
    same_number_streak = 1
    for i in range(1, min(20, len(numbers))):
        if numbers[i] == numbers[0]:
            same_number_streak += 1
        else:
            break

    if same_number_streak >= 4:
        indicators.append({
            "type": "same_number_streak",
            "level": "ANOMALY",
            "description": f"Same number {numbers[0]} appeared {same_number_streak} times consecutively",
            "value": same_number_streak,
            "threshold": 4,
        })
        overall_status = "ANOMALY"

    return {
        "status": overall_status,
        "indicators": indicators,
        "total_analyzed": len(rows),
        "current_streak": {
            "size": current_size,
            "length": current_streak,
        },
        "average_streak_length": round(avg_streak, 2),
        "small_percentage": round(small_pct, 2),
        "big_percentage": round(100 - small_pct, 2),
        "label": "HISTORICAL STATISTICS",
    }
