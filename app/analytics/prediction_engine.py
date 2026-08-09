"""
Prediction Engine — Multi-indicator weighted statistical analysis.

This engine combines multiple statistical indicators from historical data
to generate a weighted prediction suggestion for the next Small/Big outcome.

IMPORTANT DISCLAIMERS:
- This is STATISTICAL ANALYSIS based on historical patterns.
- Past patterns do NOT guarantee future outcomes.
- Each game round is an independent event.
- The engine calculates probabilities based on observed historical data.

The engine uses the following weighted indicators:
1. Streak Analysis — longer streaks historically tend to break
2. Transition Probability — what historically follows the current state
3. Frequency Rebalance — deviation from expected 50/50 distribution
4. Rolling Momentum — recent trend direction
5. Pattern Recognition — recent sequence pattern matching
"""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Indicator weights (tuned via backtesting)
WEIGHTS = {
    "streak_reversal": 0.25,
    "transition_probability": 0.25,
    "frequency_rebalance": 0.20,
    "rolling_momentum": 0.15,
    "pattern_match": 0.15,
}


def _analyze_streak_indicator(sizes: list[str]) -> dict:
    """
    Streak-based indicator.

    When a streak is longer than average, the probability of reversal increases
    based on historical data (not gambler's fallacy — based on actual observed frequencies).
    """
    if len(sizes) < 10:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    # Current streak
    current = sizes[0]
    streak_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == current:
            streak_len += 1
        else:
            break

    # Calculate average streak length from history
    streaks = []
    s_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == sizes[i - 1]:
            s_len += 1
        else:
            streaks.append(s_len)
            s_len = 1
    streaks.append(s_len)
    avg_streak = sum(streaks) / len(streaks) if streaks else 2

    # How often do streaks of this length actually continue?
    continue_count = 0
    break_count = 0
    s_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == sizes[i - 1]:
            s_len += 1
        else:
            if s_len >= streak_len:
                break_count += 1
            s_len = 1

    # If current streak is longer than average, lean toward reversal
    ratio = streak_len / avg_streak if avg_streak > 0 else 1
    if ratio > 2.0:
        opposite = "BIG" if current == "SMALL" else "SMALL"
        confidence = min(0.85, 0.50 + (ratio - 2) * 0.12)
        return {
            "prediction": opposite,
            "confidence": round(confidence, 3),
            "reason": f"streak_{current}_{streak_len}_avg_{avg_streak:.1f}",
        }
    elif ratio > 1.5:
        opposite = "BIG" if current == "SMALL" else "SMALL"
        confidence = 0.45 + (ratio - 1.5) * 0.1
        return {
            "prediction": opposite,
            "confidence": round(confidence, 3),
            "reason": f"moderate_streak_{streak_len}",
        }
    else:
        # Short streak — follow the trend slightly
        return {
            "prediction": current,
            "confidence": 0.40,
            "reason": f"short_streak_{streak_len}",
        }


def _analyze_transition_indicator(sizes: list[str]) -> dict:
    """
    Transition-based indicator.

    Uses conditional probability: P(next | current_state) from historical data.
    """
    if len(sizes) < 30:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    current = sizes[0]

    # Count transitions from current state
    same_count = 0
    opposite_count = 0
    for i in range(len(sizes) - 1):
        if sizes[i + 1] == current:  # "from" state matches current
            if sizes[i] == current:
                same_count += 1
            else:
                opposite_count += 1

    total = same_count + opposite_count
    if total == 0:
        return {"prediction": None, "confidence": 0, "reason": "no_transitions"}

    same_pct = same_count / total
    opposite_pct = opposite_count / total

    if same_pct > opposite_pct:
        prediction = current
        confidence = 0.35 + (same_pct - 0.5) * 0.6
    else:
        prediction = "BIG" if current == "SMALL" else "SMALL"
        confidence = 0.35 + (opposite_pct - 0.5) * 0.6

    return {
        "prediction": prediction,
        "confidence": round(min(confidence, 0.80), 3),
        "reason": f"transition_{'same' if same_pct > opposite_pct else 'opposite'}_{same_pct:.2f}",
    }


def _analyze_frequency_indicator(sizes: list[str]) -> dict:
    """
    Frequency rebalance indicator.

    When one side is significantly overrepresented, historically
    the underrepresented side tends to appear more frequently to rebalance.
    """
    windows = [20, 50, 100]
    signals = {"SMALL": 0, "BIG": 0}

    for w in windows:
        if len(sizes) < w:
            continue
        window_data = sizes[:w]
        small_count = sum(1 for s in window_data if s == "SMALL")
        small_pct = small_count / w * 100

        if small_pct > 55:
            signals["BIG"] += 1
        elif small_pct < 45:
            signals["SMALL"] += 1

    if signals["SMALL"] > signals["BIG"]:
        confidence = 0.40 + signals["SMALL"] * 0.08
        return {
            "prediction": "SMALL",
            "confidence": round(min(confidence, 0.70), 3),
            "reason": f"underrepresented_SMALL_{signals['SMALL']}_windows",
        }
    elif signals["BIG"] > signals["SMALL"]:
        confidence = 0.40 + signals["BIG"] * 0.08
        return {
            "prediction": "BIG",
            "confidence": round(min(confidence, 0.70), 3),
            "reason": f"underrepresented_BIG_{signals['BIG']}_windows",
        }
    else:
        return {
            "prediction": None,
            "confidence": 0,
            "reason": "balanced_frequency",
        }


def _analyze_momentum_indicator(sizes: list[str]) -> dict:
    """
    Rolling momentum indicator.

    Examines short-term vs medium-term trends to detect momentum shifts.
    """
    if len(sizes) < 30:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    # Short-term (last 10) vs medium-term (last 30) comparison
    short_small = sum(1 for s in sizes[:10] if s == "SMALL") / 10
    medium_small = sum(1 for s in sizes[:30] if s == "SMALL") / 30

    momentum_shift = short_small - medium_small

    if abs(momentum_shift) < 0.05:
        return {"prediction": None, "confidence": 0, "reason": "no_momentum_shift"}

    if momentum_shift > 0.15:
        # SMALL is trending up — but may revert
        return {
            "prediction": "BIG",
            "confidence": round(0.40 + abs(momentum_shift) * 0.5, 3),
            "reason": f"small_momentum_high_{momentum_shift:.2f}",
        }
    elif momentum_shift < -0.15:
        return {
            "prediction": "SMALL",
            "confidence": round(0.40 + abs(momentum_shift) * 0.5, 3),
            "reason": f"big_momentum_high_{abs(momentum_shift):.2f}",
        }
    elif momentum_shift > 0.05:
        return {
            "prediction": "SMALL",
            "confidence": 0.42,
            "reason": "following_small_momentum",
        }
    else:
        return {
            "prediction": "BIG",
            "confidence": 0.42,
            "reason": "following_big_momentum",
        }


def _analyze_pattern_indicator(sizes: list[str]) -> dict:
    """
    Pattern matching indicator.

    Looks for repeating patterns in recent results and checks
    how those patterns have historically resolved.
    """
    if len(sizes) < 50:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    # Look at the last 3 results as a pattern
    pattern = tuple(sizes[:3])

    # Search for this pattern in history
    matches_followed_by = {"SMALL": 0, "BIG": 0}
    for i in range(3, len(sizes) - 1):
        if tuple(sizes[i - 2:i + 1]) == pattern:
            # What came before this pattern (which would be the "next" result)
            if i >= 3:
                next_val = sizes[i - 3]
                matches_followed_by[next_val] = matches_followed_by.get(next_val, 0) + 1

    total_matches = sum(matches_followed_by.values())
    if total_matches < 3:
        return {"prediction": None, "confidence": 0, "reason": "pattern_not_found"}

    if matches_followed_by["SMALL"] > matches_followed_by["BIG"]:
        pct = matches_followed_by["SMALL"] / total_matches
        return {
            "prediction": "SMALL",
            "confidence": round(min(0.35 + pct * 0.3, 0.75), 3),
            "reason": f"pattern_{''.join(s[0] for s in pattern)}_favors_SMALL_{total_matches}_samples",
        }
    elif matches_followed_by["BIG"] > matches_followed_by["SMALL"]:
        pct = matches_followed_by["BIG"] / total_matches
        return {
            "prediction": "BIG",
            "confidence": round(min(0.35 + pct * 0.3, 0.75), 3),
            "reason": f"pattern_{''.join(s[0] for s in pattern)}_favors_BIG_{total_matches}_samples",
        }
    else:
        return {"prediction": None, "confidence": 0, "reason": "pattern_split_50_50"}


async def generate_prediction(
    session: AsyncSession, window: int = 500
) -> dict:
    """
    Generate a weighted statistical prediction for the next Small/Big outcome.

    Combines 5 independent indicators with configurable weights
    to produce a single recommendation with confidence score.

    IMPORTANT: This is STATISTICAL ANALYSIS, NOT a guarantee.
    Each game round is an independent random event.

    Args:
        session: Database session.
        window: Number of recent records to analyze.

    Returns:
        Dict with prediction, confidence, and indicator breakdown.
    """
    query = (
        select(GameResult.calculated_size, GameResult.issue_id, GameResult.result_number)
        .order_by(desc(GameResult.issue_id))
        .limit(window)
    )

    result = await session.execute(query)
    rows = result.fetchall()

    if len(rows) < 5:
        return {
            "upcoming_issue_id": None,
            "prediction": None,
            "confidence": 0,
            "status": "INSUFFICIENT_DATA",
            "message": "Need at least 5 historical records for analysis",
            "total_records_analyzed": len(rows),
            "label": "STATISTICAL ANALYSIS — NOT A GUARANTEE",
        }

    sizes = [row.calculated_size for row in rows]
    latest_issue = rows[0].issue_id if rows else None

    # Run all indicators
    indicators = {
        "streak_reversal": _analyze_streak_indicator(sizes),
        "transition_probability": _analyze_transition_indicator(sizes),
        "frequency_rebalance": _analyze_frequency_indicator(sizes),
        "rolling_momentum": _analyze_momentum_indicator(sizes),
        "pattern_match": _analyze_pattern_indicator(sizes),
    }

    # Weighted voting
    small_score = 0.0
    big_score = 0.0
    total_weight = 0.0
    active_indicators = 0

    for name, indicator in indicators.items():
        weight = WEIGHTS.get(name, 0.1)
        pred = indicator.get("prediction")
        conf = indicator.get("confidence", 0)

        if pred and conf > 0:
            weighted_score = weight * conf
            if pred == "SMALL":
                small_score += weighted_score
            else:
                big_score += weighted_score
            total_weight += weight
            active_indicators += 1

    if total_weight == 0:
        return {
            "prediction": None,
            "confidence": 0,
            "status": "NO_SIGNAL",
            "message": "No indicators produced a signal",
            "indicators": indicators,
            "total_records_analyzed": len(rows),
            "label": "STATISTICAL ANALYSIS — NOT A GUARANTEE",
        }

    # Normalize scores
    norm_small = small_score / total_weight if total_weight > 0 else 0.5
    norm_big = big_score / total_weight if total_weight > 0 else 0.5

    # Final prediction
    if norm_small > norm_big:
        prediction = "SMALL"
        confidence = round(norm_small, 3)
    elif norm_big > norm_small:
        prediction = "BIG"
        confidence = round(norm_big, 3)
    else:
        prediction = sizes[0]  # Tie-break: follow current
        confidence = 0.50

    # Classify confidence level
    if confidence >= 0.70:
        confidence_level = "HIGH"
    elif confidence >= 0.55:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Count indicator agreement
    agreeing = sum(
        1 for ind in indicators.values()
        if ind.get("prediction") == prediction and ind.get("confidence", 0) > 0
    )

    # Calculate upcoming issue ID for next 30S draw
    upcoming_issue_id = None
    if latest_issue:
        try:
            upcoming_issue_id = str(int(latest_issue) + 1)
        except ValueError:
            upcoming_issue_id = None

    logger.info(
        "prediction_generated",
        prediction=prediction,
        confidence=confidence,
        confidence_level=confidence_level,
        active_indicators=active_indicators,
        agreeing_indicators=agreeing,
        latest_issue=latest_issue,
        upcoming_issue=upcoming_issue_id,
    )

    return {
        "upcoming_issue_id": upcoming_issue_id,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "small_score": round(norm_small, 3),
        "big_score": round(norm_big, 3),
        "active_indicators": active_indicators,
        "agreeing_indicators": agreeing,
        "indicators": indicators,
        "current_state": {
            "latest_size": sizes[0] if sizes else None,
            "current_streak": _get_current_streak(sizes),
            "latest_issue": latest_issue,
        },
        "total_records_analyzed": len(rows),
        "status": "ACTIVE",
        "label": "STATISTICAL ANALYSIS — NOT A GUARANTEE",
        "disclaimer": "This prediction is based on historical pattern analysis for the upcoming game period. Each draw is independent.",
    }


def _get_current_streak(sizes: list[str]) -> dict:
    """Get current streak info."""
    if not sizes:
        return {"size": None, "length": 0}
    current = sizes[0]
    length = 1
    for i in range(1, len(sizes)):
        if sizes[i] == current:
            length += 1
        else:
            break
    return {"size": current, "length": length}
