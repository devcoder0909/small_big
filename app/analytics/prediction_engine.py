"""
Prediction Engine — Unstoppable Multi-Indicator Weighted Statistical Analysis.

This engine combines multiple advanced statistical and machine learning inspired indicators:
1. Empirical Streak Reversal & Odds Analysis
2. Multi-Order Markov Chain State Transitions (Orders 1, 2, and 3)
3. Z-Score & Statistical Significance Frequency Rebalance
4. Dual Exponential Moving Average (EMA) Momentum Crossover
5. Variable N-Gram Multi-Length Pattern Recognition (Lengths 2 to 5)
6. Shannon Entropy & Regime Shift Noise Filter

IMPORTANT DISCLAIMERS:
- This is STATISTICAL ANALYSIS based on historical patterns.
- Past patterns do NOT guarantee future outcomes.
- Each game round is an independent random event.
- The engine calculates probabilities based on observed historical data.
"""

import math
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Base indicator weights
DEFAULT_WEIGHTS = {
    "streak_reversal": 0.22,
    "markov_transition": 0.24,
    "stat_frequency": 0.20,
    "ema_momentum": 0.16,
    "pattern_match": 0.18,
}


def _calculate_shannon_entropy(sizes: list[str]) -> float:
    """
    Calculate binary Shannon Entropy of the sequence.
    H(X) = - p(Small) * log2(p(Small)) - p(Big) * log2(p(Big))
    Range: 0.0 (fully deterministic) to 1.0 (pure uniform random).
    """
    if not sizes:
        return 1.0
    total = len(sizes)
    small_count = sum(1 for s in sizes if s == "SMALL")
    p_small = small_count / total
    p_big = 1.0 - p_small

    if p_small == 0 or p_big == 0:
        return 0.0

    entropy = - (p_small * math.log2(p_small) + p_big * math.log2(p_big))
    return round(entropy, 4)


def _calculate_z_score(sizes: list[str]) -> tuple[float, float]:
    """
    Calculate Z-score for frequency deviation from expected 50% ratio.
    Z = (x - n*p) / sqrt(n*p*(1-p))
    Returns (z_score, p_value_approx).
    """
    n = len(sizes)
    if n == 0:
        return 0.0, 1.0
    small_count = sum(1 for s in sizes if s == "SMALL")
    expected_mean = n * 0.5
    std_dev = math.sqrt(n * 0.5 * 0.5)

    if std_dev == 0:
        return 0.0, 1.0

    z = (small_count - expected_mean) / std_dev
    return round(z, 3), round(small_count / n, 4)


def _analyze_streak_indicator(sizes: list[str]) -> dict:
    """
    Empirical streak analysis.
    Calculates actual historical break vs continuation frequencies for streaks of the current length.
    """
    if len(sizes) < 10:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    current = sizes[0]
    streak_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == current:
            streak_len += 1
        else:
            break

    # Calculate historical streaks and their outcomes
    continue_count = 0
    break_count = 0

    s_len = 1
    for i in range(len(sizes) - 2, -1, -1):
        if sizes[i] == sizes[i + 1]:
            s_len += 1
        else:
            if s_len == streak_len:
                break_count += 1
            elif s_len > streak_len:
                continue_count += 1
            s_len = 1

    total_observed = continue_count + break_count
    opposite = "BIG" if current == "SMALL" else "SMALL"

    if total_observed >= 5:
        break_ratio = break_count / total_observed
        if break_ratio > 0.55:
            confidence = min(0.85, 0.50 + (break_ratio - 0.55) * 0.8)
            return {
                "prediction": opposite,
                "confidence": round(confidence, 3),
                "reason": f"empirical_streak_reversal_ratio_{break_ratio:.2f}_n_{total_observed}",
            }
        elif break_ratio < 0.45:
            confidence = min(0.80, 0.50 + (0.45 - break_ratio) * 0.8)
            return {
                "prediction": current,
                "confidence": round(confidence, 3),
                "reason": f"empirical_streak_continuation_ratio_{1-break_ratio:.2f}_n_{total_observed}",
            }

    # Fallback to streak ratio heuristic if sample size small
    streaks = []
    s_len = 1
    for i in range(1, len(sizes)):
        if sizes[i] == sizes[i - 1]:
            s_len += 1
        else:
            streaks.append(s_len)
            s_len = 1
    streaks.append(s_len)
    avg_streak = sum(streaks) / len(streaks) if streaks else 2.0

    ratio = streak_len / avg_streak if avg_streak > 0 else 1.0
    if ratio >= 2.0:
        confidence = min(0.82, 0.50 + (ratio - 2.0) * 0.15)
        return {
            "prediction": opposite,
            "confidence": round(confidence, 3),
            "reason": f"streak_len_{streak_len}_exceeds_avg_{avg_streak:.1f}",
        }
    else:
        return {
            "prediction": current,
            "confidence": 0.45,
            "reason": f"short_streak_{streak_len}",
        }


def _analyze_markov_transition_indicator(sizes: list[str]) -> dict:
    """
    Multi-Order Markov Chain transition probability (Order 1, Order 2, Order 3).
    P(next | state_t-1, state_t-2, state_t-3)
    """
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    # We evaluate Order 3, Order 2, Order 1 transitions
    scores = {"SMALL": 0.0, "BIG": 0.0}
    weights = {3: 0.5, 2: 0.3, 1: 0.2}
    details = []

    for order in (3, 2, 1):
        if len(sizes) <= order:
            continue
        context = tuple(sizes[:order])
        same_next = 0
        opp_next = 0

        for i in range(order, len(sizes) - 1):
            if tuple(sizes[i - order + 1 : i + 1]) == context:
                next_item = sizes[i - order]
                if next_item == "SMALL":
                    same_next += 1
                else:
                    opp_next += 1

        total = same_next + opp_next
        if total >= 3:
            s_pct = same_next / total
            b_pct = opp_next / total
            scores["SMALL"] += s_pct * weights[order]
            scores["BIG"] += b_pct * weights[order]
            details.append(f"O{order}:{s_pct:.2f}/{b_pct:.2f}(n={total})")

    if not details:
        return {"prediction": None, "confidence": 0, "reason": "no_markov_history"}

    if scores["SMALL"] > scores["BIG"]:
        conf = 0.40 + min(0.40, (scores["SMALL"] - scores["BIG"]) * 0.8)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"markov_{'_'.join(details)}",
        }
    elif scores["BIG"] > scores["SMALL"]:
        conf = 0.40 + min(0.40, (scores["BIG"] - scores["SMALL"]) * 0.8)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"markov_{'_'.join(details)}",
        }
    else:
        return {"prediction": None, "confidence": 0, "reason": "markov_balanced"}


def _analyze_statistical_frequency_indicator(sizes: list[str]) -> dict:
    """
    Frequency rebalance indicator using Z-Score statistical significance.
    """
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    z_20, p_20 = _calculate_z_score(sizes[:20])
    z_50, p_50 = _calculate_z_score(sizes[:min(50, len(sizes))])
    z_100, p_100 = _calculate_z_score(sizes[:min(100, len(sizes))])

    # Combine Z-scores
    composite_z = z_20 * 0.5 + z_50 * 0.3 + z_100 * 0.2

    # Significant overrepresentation of SMALL -> predict BIG
    if composite_z > 1.5:
        conf = min(0.85, 0.45 + (composite_z - 1.5) * 0.15)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"stat_rebalance_z_{composite_z:.2f}_small_overrepresented",
        }
    # Significant overrepresentation of BIG -> predict SMALL
    elif composite_z < -1.5:
        conf = min(0.85, 0.45 + (abs(composite_z) - 1.5) * 0.15)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"stat_rebalance_z_{composite_z:.2f}_big_overrepresented",
        }
    else:
        return {
            "prediction": None,
            "confidence": 0,
            "reason": f"frequency_in_normal_range_z_{composite_z:.2f}",
        }


def _analyze_ema_momentum_indicator(sizes: list[str]) -> dict:
    """
    Dual Exponential Moving Average (EMA) momentum indicator.
    Calculates EMA(5) vs EMA(20) to identify micro trend momentum shifts.
    """
    if len(sizes) < 25:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    numeric = [1.0 if s == "SMALL" else 0.0 for s in reversed(sizes[:30])]

    # Calculate EMA
    def _ema(values: list[float], span: int) -> float:
        alpha = 2.0 / (span + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    fast_ema = _ema(numeric, 5)
    slow_ema = _ema(numeric, 20)
    diff = fast_ema - slow_ema

    if abs(diff) < 0.05:
        return {"prediction": None, "confidence": 0, "reason": "no_momentum_cross"}

    if diff > 0.15:
        # Fast SMALL momentum high -> mean reversion to BIG
        return {
            "prediction": "BIG",
            "confidence": round(min(0.78, 0.45 + diff * 0.6), 3),
            "reason": f"ema_diff_small_high_{diff:.2f}",
        }
    elif diff < -0.15:
        # Fast BIG momentum high -> mean reversion to SMALL
        return {
            "prediction": "SMALL",
            "confidence": round(min(0.78, 0.45 + abs(diff) * 0.6), 3),
            "reason": f"ema_diff_big_high_{abs(diff):.2f}",
        }
    elif diff > 0:
        return {
            "prediction": "SMALL",
            "confidence": 0.44,
            "reason": "following_small_momentum",
        }
    else:
        return {
            "prediction": "BIG",
            "confidence": 0.44,
            "reason": "following_big_momentum",
        }


def _analyze_multi_ngram_pattern_indicator(sizes: list[str]) -> dict:
    """
    Multi-length N-gram pattern recognition engine (N = 2, 3, 4, 5).
    Searches historical sequence for matching sequence tails.
    """
    if len(sizes) < 40:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    small_votes = 0.0
    big_votes = 0.0
    pattern_matches_info = []

    for n in (5, 4, 3, 2):
        if len(sizes) <= n:
            continue
        pattern = tuple(sizes[:n])
        match_small = 0
        match_big = 0

        for i in range(n, len(sizes)):
            if tuple(sizes[i - n + 1 : i + 1]) == pattern and i - n >= 0:
                next_val = sizes[i - n]
                if next_val == "SMALL":
                    match_small += 1
                elif next_val == "BIG":
                    match_big += 1

        total = match_small + match_big
        if total >= 2:
            weight = n * 0.25
            small_votes += (match_small / total) * weight
            big_votes += (match_big / total) * weight
            pattern_matches_info.append(f"N{n}:S{match_small}/B{match_big}")

    total_votes = small_votes + big_votes
    if total_votes == 0:
        return {"prediction": None, "confidence": 0, "reason": "no_pattern_matches"}

    norm_small = small_votes / total_votes
    norm_big = big_votes / total_votes

    if norm_small > norm_big:
        conf = min(0.82, 0.40 + (norm_small - 0.5) * 0.8)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"ngram_pattern_matches_{','.join(pattern_matches_info)}",
        }
    elif norm_big > norm_small:
        conf = min(0.82, 0.40 + (norm_big - 0.5) * 0.8)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"ngram_pattern_matches_{','.join(pattern_matches_info)}",
        }
    else:
        return {"prediction": None, "confidence": 0, "reason": "ngram_split_50_50"}


async def generate_prediction(
    session: AsyncSession, window: int = 500
) -> dict:
    """
    Generate an advanced statistical prediction for the upcoming WinGo 30S draw.

    Combines 5 independent indicators with adaptive dynamic weighting:
    - Empirical Streak Reversal
    - Multi-Order Markov Chain Transitions
    - Z-Score Statistical Frequency Rebalance
    - Dual EMA Momentum Crossover
    - Multi-Length N-Gram Pattern Matching

    Includes Shannon Entropy measurement for sequence noise filtering.

    Args:
        session: Active database session.
        window: Number of recent records to analyze.

    Returns:
        Dict with prediction, confidence, entropy, and indicator breakdown.
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

    # Calculate sequence Shannon Entropy & Z-Score
    shannon_entropy = _calculate_shannon_entropy(sizes[:50])
    z_score, p_small = _calculate_z_score(sizes)

    # Run all 5 indicators
    indicators = {
        "streak_reversal": _analyze_streak_indicator(sizes),
        "markov_transition": _analyze_markov_transition_indicator(sizes),
        "stat_frequency": _analyze_statistical_frequency_indicator(sizes),
        "ema_momentum": _analyze_ema_momentum_indicator(sizes),
        "pattern_match": _analyze_multi_ngram_pattern_indicator(sizes),
    }

    # Adaptive Dynamic Weighting based on entropy
    # If entropy < 0.95 (regime shift), boost pattern and markov weights; if high noise, boost frequency rebalance
    weights = dict(DEFAULT_WEIGHTS)
    if shannon_entropy < 0.90:
        weights["markov_transition"] += 0.05
        weights["pattern_match"] += 0.05
        weights["ema_momentum"] -= 0.05
        weights["stat_frequency"] -= 0.05

    # Weighted voting
    small_score = 0.0
    big_score = 0.0
    total_weight = 0.0
    active_indicators = 0

    for name, indicator in indicators.items():
        w = weights.get(name, 0.20)
        pred = indicator.get("prediction")
        conf = indicator.get("confidence", 0)

        if pred and conf > 0:
            weighted_val = w * conf
            if pred == "SMALL":
                small_score += weighted_val
            else:
                big_score += weighted_val
            total_weight += w
            active_indicators += 1

    if total_weight == 0:
        return {
            "prediction": None,
            "confidence": 0,
            "status": "NO_SIGNAL",
            "message": "No indicators produced a signal",
            "shannon_entropy": shannon_entropy,
            "z_score": z_score,
            "indicators": indicators,
            "total_records_analyzed": len(rows),
            "label": "STATISTICAL ANALYSIS — NOT A GUARANTEE",
        }

    # Normalize scores
    norm_small = small_score / total_weight if total_weight > 0 else 0.5
    norm_big = big_score / total_weight if total_weight > 0 else 0.5

    # Final decision
    if norm_small > norm_big:
        prediction = "SMALL"
        confidence = round(norm_small, 3)
    elif norm_big > norm_small:
        prediction = "BIG"
        confidence = round(norm_big, 3)
    else:
        prediction = sizes[0]  # Tie-break: follow latest
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

    # Calculate upcoming issue ID
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
        entropy=shannon_entropy,
        z_score=z_score,
        active_indicators=active_indicators,
        agreeing_indicators=agreeing,
        upcoming_issue=upcoming_issue_id,
    )

    return {
        "upcoming_issue_id": upcoming_issue_id,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "shannon_entropy": shannon_entropy,
        "z_score": z_score,
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
        "disclaimer": "This prediction is based on historical statistical analysis for the upcoming game period. Each draw is an independent random event.",
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
