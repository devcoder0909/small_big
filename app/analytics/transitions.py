"""Transition analysis — historical transition matrix between Small and Big."""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


async def calculate_transitions(
    session: AsyncSession, window: int | None = None
) -> dict:
    """
    Calculate historical transition frequencies.

    HISTORICAL STATISTICS — past transitions do NOT determine future outcomes.

    Transitions tracked:
        SMALL → SMALL
        SMALL → BIG
        BIG → SMALL
        BIG → BIG

    Args:
        session: Database session.
        window: Number of recent records. None = all-time.

    Returns:
        Dict with transition counts and percentages.
    """
    query = (
        select(GameResult.calculated_size)
        .order_by(desc(GameResult.issue_id))
    )
    if window:
        query = query.limit(window)

    result = await session.execute(query)
    rows = result.fetchall()

    if len(rows) < 2:
        return {
            "total_transitions": 0,
            "transitions": {},
            "percentages": {},
            "label": "HISTORICAL STATISTICS",
        }

    sizes = [row.calculated_size for row in rows]

    # Count transitions (newest-first, so sizes[0]→sizes[1] is latest transition)
    transitions = {
        "SMALL_to_SMALL": 0,
        "SMALL_to_BIG": 0,
        "BIG_to_SMALL": 0,
        "BIG_to_BIG": 0,
    }

    for i in range(len(sizes) - 1):
        current = sizes[i + 1]  # older (from)
        next_val = sizes[i]     # newer (to)
        key = f"{current}_to_{next_val}"
        transitions[key] = transitions.get(key, 0) + 1

    total = sum(transitions.values())

    percentages = {
        k: round(v / total * 100, 2) if total > 0 else 0
        for k, v in transitions.items()
    }

    # Conditional probabilities
    small_total = transitions["SMALL_to_SMALL"] + transitions["SMALL_to_BIG"]
    big_total = transitions["BIG_to_SMALL"] + transitions["BIG_to_BIG"]

    conditional = {
        "after_SMALL": {
            "SMALL": round(transitions["SMALL_to_SMALL"] / small_total * 100, 2) if small_total > 0 else 50.0,
            "BIG": round(transitions["SMALL_to_BIG"] / small_total * 100, 2) if small_total > 0 else 50.0,
        },
        "after_BIG": {
            "SMALL": round(transitions["BIG_to_SMALL"] / big_total * 100, 2) if big_total > 0 else 50.0,
            "BIG": round(transitions["BIG_to_BIG"] / big_total * 100, 2) if big_total > 0 else 50.0,
        },
    }

    return {
        "total_transitions": total,
        "transitions": transitions,
        "percentages": percentages,
        "conditional_probabilities": conditional,
        "label": "HISTORICAL STATISTICS",
    }
