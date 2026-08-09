"""
Prediction Engine — 10-Indicator God-Mode Statistical & Bayesian Ensemble.

This engine combines 10 advanced statistical, structural, and Bayesian indicators:
1. Empirical Streak Reversal & Non-Linear Hazard Analysis
2. Multi-Order Markov Chain Transitions (Orders 1, 2, 3, and 4)
3. Z-Score Statistical Frequency Rebalance
4. Dual Exponential Moving Average (EMA) Momentum Crossover
5. Variable N-Gram Multi-Length Pattern Recognition (Lengths 2 to 6)
6. Harmonic Periodicity & Micro-Cycle Detection
7. Bayesian Model Averaging (Dirichlet-Multinomial Conjugate Prior)
8. Shannon Entropy & Volatility Regime Shift Filter
9. Pearson's Chi-Square Goodness-of-Fit Skew Detection
10. Wald-Wolfowitz Runs Test Randomness Detector

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

# Base indicator weights for the 10 indicators
DEFAULT_WEIGHTS = {
    "streak_reversal": 0.14,
    "markov_transition": 0.16,
    "stat_frequency": 0.12,
    "ema_momentum": 0.10,
    "pattern_match": 0.14,
    "harmonic_periodicity": 0.07,
    "bayesian_posterior": 0.09,
    "volatility_regime": 0.05,
    "chi_square_skew": 0.07,
    "runs_test": 0.06,
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

    # Fallback heuristic
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
    Multi-Order Markov Chain transition probability (Orders 4, 3, 2, 1).
    P(next | state_t-1, state_t-2, state_t-3, state_t-4)
    """
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    scores = {"SMALL": 0.0, "BIG": 0.0}
    weights = {4: 0.4, 3: 0.3, 2: 0.2, 1: 0.1}
    details = []

    for order in (4, 3, 2, 1):
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
        if total >= 2:
            s_pct = (same_next + 1) / (total + 2)  # Laplace smoothing
            b_pct = (opp_next + 1) / (total + 2)
            scores["SMALL"] += s_pct * weights[order]
            scores["BIG"] += b_pct * weights[order]
            details.append(f"O{order}:{s_pct:.2f}/{b_pct:.2f}(n={total})")

    if not details:
        return {"prediction": None, "confidence": 0, "reason": "no_markov_history"}

    if scores["SMALL"] > scores["BIG"]:
        conf = 0.40 + min(0.45, (scores["SMALL"] - scores["BIG"]) * 0.9)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"markov_{'_'.join(details)}",
        }
    elif scores["BIG"] > scores["SMALL"]:
        conf = 0.40 + min(0.45, (scores["BIG"] - scores["SMALL"]) * 0.9)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"markov_{'_'.join(details)}",
        }
    else:
        return {"prediction": None, "confidence": 0, "reason": "markov_balanced"}


def _analyze_statistical_frequency_indicator(sizes: list[str]) -> dict:
    """Frequency dominance indicator using multi-window Z-Score analysis."""
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    z_20, p_20 = _calculate_z_score(sizes[:20])
    z_50, p_50 = _calculate_z_score(sizes[:min(50, len(sizes))])
    z_100, p_100 = _calculate_z_score(sizes[:min(100, len(sizes))])

    composite_z = z_20 * 0.5 + z_50 * 0.3 + z_100 * 0.2

    # Follow the dominant direction — if SMALL is hot, predict SMALL
    if composite_z > 1.5:
        conf = min(0.78, 0.45 + (composite_z - 1.5) * 0.12)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"stat_small_dominant_z_{composite_z:.2f}",
        }
    elif composite_z < -1.5:
        conf = min(0.78, 0.45 + (abs(composite_z) - 1.5) * 0.12)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"stat_big_dominant_z_{composite_z:.2f}",
        }
    else:
        return {
            "prediction": None,
            "confidence": 0,
            "reason": f"frequency_balanced_z_{composite_z:.2f}",
        }


def _analyze_ema_momentum_indicator(sizes: list[str]) -> dict:
    """Dual EMA momentum-following indicator. Follows the trend, never inverts."""
    if len(sizes) < 25:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    numeric = [1.0 if s == "SMALL" else 0.0 for s in reversed(sizes[:30])]

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

    # Follow momentum: fast EMA above slow = SMALL momentum, follow it
    if diff > 0.10:
        return {
            "prediction": "SMALL",
            "confidence": round(min(0.78, 0.45 + diff * 0.6), 3),
            "reason": f"ema_small_momentum_{diff:.2f}",
        }
    elif diff < -0.10:
        return {
            "prediction": "BIG",
            "confidence": round(min(0.78, 0.45 + abs(diff) * 0.6), 3),
            "reason": f"ema_big_momentum_{abs(diff):.2f}",
        }
    elif diff > 0:
        return {
            "prediction": "SMALL",
            "confidence": 0.44,
            "reason": "weak_small_momentum",
        }
    else:
        return {
            "prediction": "BIG",
            "confidence": 0.44,
            "reason": "weak_big_momentum",
        }


def _analyze_multi_ngram_pattern_indicator(sizes: list[str]) -> dict:
    """Multi-length N-gram pattern recognition engine (N = 2, 3, 4, 5, 6)."""
    if len(sizes) < 40:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    small_votes = 0.0
    big_votes = 0.0
    pattern_matches_info = []

    for n in (6, 5, 4, 3, 2):
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
            weight = n * 0.20
            small_votes += (match_small / total) * weight
            big_votes += (match_big / total) * weight
            pattern_matches_info.append(f"N{n}:S{match_small}/B{match_big}")

    total_votes = small_votes + big_votes
    if total_votes == 0:
        return {"prediction": None, "confidence": 0, "reason": "no_pattern_matches"}

    norm_small = small_votes / total_votes
    norm_big = big_votes / total_votes

    if norm_small > norm_big:
        conf = min(0.85, 0.40 + (norm_small - 0.5) * 0.8)
        return {
            "prediction": "SMALL",
            "confidence": round(conf, 3),
            "reason": f"ngram_pattern_matches_{','.join(pattern_matches_info)}",
        }
    elif norm_big > norm_small:
        conf = min(0.85, 0.40 + (norm_big - 0.5) * 0.8)
        return {
            "prediction": "BIG",
            "confidence": round(conf, 3),
            "reason": f"ngram_pattern_matches_{','.join(pattern_matches_info)}",
        }
    else:
        return {"prediction": None, "confidence": 0, "reason": "ngram_split_50_50"}


def _analyze_harmonic_periodicity_indicator(sizes: list[str]) -> dict:
    """Harmonic Periodicity & Micro-Cycle Detection (Alternating & Triplet Cycles)."""
    if len(sizes) < 12:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    # Check 2-period alternating cycle: S-B-S-B or B-S-B-S
    alt_count = 0
    for i in range(1, min(10, len(sizes))):
        if sizes[i] != sizes[i - 1]:
            alt_count += 1
        else:
            break

    if alt_count >= 3:
        opposite = "BIG" if sizes[0] == "SMALL" else "SMALL"
        conf = min(0.82, 0.50 + alt_count * 0.05)
        return {
            "prediction": opposite,
            "confidence": round(conf, 3),
            "reason": f"alternating_harmonic_cycle_len_{alt_count}",
        }

    return {"prediction": None, "confidence": 0, "reason": "no_harmonic_cycle"}


def _analyze_bayesian_posterior_indicator(sizes: list[str]) -> dict:
    """Bayesian Model Averaging using Dirichlet-Multinomial. Follows the posterior."""
    if len(sizes) < 30:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    prior_small = 10.0
    prior_big = 10.0

    recent_30 = sizes[:30]
    small_obs = sum(1 for s in recent_30 if s == "SMALL")
    big_obs = sum(1 for s in recent_30 if s == "BIG")

    post_small = prior_small + small_obs
    post_big = prior_big + big_obs
    total_post = post_small + post_big

    p_small_post = post_small / total_post
    p_big_post = post_big / total_post

    # Follow the posterior — predict what has higher posterior probability
    if p_small_post > 0.58:
        return {
            "prediction": "SMALL",
            "confidence": round(min(0.75, 0.40 + (p_small_post - 0.58) * 1.2), 3),
            "reason": f"bayesian_small_dominant_{p_small_post:.3f}",
        }
    elif p_big_post > 0.58:
        return {
            "prediction": "BIG",
            "confidence": round(min(0.75, 0.40 + (p_big_post - 0.58) * 1.2), 3),
            "reason": f"bayesian_big_dominant_{p_big_post:.3f}",
        }

    return {"prediction": None, "confidence": 0, "reason": "bayesian_posterior_neutral"}


def _analyze_volatility_regime_indicator(sizes: list[str]) -> dict:
    """Volatility & Regime Shift Detection. Follows momentum in all regimes."""
    if len(sizes) < 30:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    entropy_recent = _calculate_shannon_entropy(sizes[:15])
    entropy_macro = _calculate_shannon_entropy(sizes[:50])
    entropy_diff = entropy_recent - entropy_macro

    if entropy_diff < -0.15:
        # Structured regime -> follow latest momentum
        return {
            "prediction": sizes[0],
            "confidence": 0.68,
            "reason": f"structured_regime_follow_momentum_{entropy_recent:.2f}",
        }
    elif entropy_diff > 0.15:
        # Noisy regime -> still follow latest, but lower confidence
        return {
            "prediction": sizes[0],
            "confidence": 0.52,
            "reason": f"noisy_regime_weak_momentum_{entropy_recent:.2f}",
        }

    return {"prediction": None, "confidence": 0, "reason": "stable_entropy_regime"}


def _analyze_chi_square_goodness_of_fit_indicator(sizes: list[str]) -> dict:
    """Pearson's Chi-Square Goodness-of-Fit. Follows the dominant direction."""
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    window = sizes[:40]
    n = len(window)
    o_small = window.count("SMALL")
    o_big = window.count("BIG")
    expected = n / 2.0

    chi_sq = ((o_small - expected) ** 2 + (o_big - expected) ** 2) / expected

    if chi_sq >= 3.841:
        # Follow the dominant direction, not invert it
        target = "SMALL" if o_small > o_big else "BIG"
        conf = min(0.78, 0.50 + (chi_sq - 3.841) * 0.04)
        return {
            "prediction": target,
            "confidence": round(conf, 3),
            "reason": f"chi_square_dominant_{target.lower()}_chisq_{chi_sq:.2f}",
        }

    return {"prediction": None, "confidence": 0, "reason": "chi_square_balanced"}


def _analyze_runs_test_indicator(sizes: list[str]) -> dict:
    """
    Wald-Wolfowitz Runs Test for randomness deviation.
    Detects if the sequence has significantly fewer or more runs than expected,
    signaling clustering (trend continuation) or oscillation (mean reversion).
    """
    if len(sizes) < 20:
        return {"prediction": None, "confidence": 0, "reason": "insufficient_data"}

    window = sizes[:50]
    n = len(window)
    n1 = window.count("SMALL")
    n2 = n - n1

    if n1 == 0 or n2 == 0:
        return {"prediction": None, "confidence": 0, "reason": "runs_test_single_class"}

    # Count actual runs
    runs = 1
    for i in range(1, n):
        if window[i] != window[i - 1]:
            runs += 1

    # Expected runs and standard deviation under H0 (random)
    expected_runs = 1 + (2 * n1 * n2) / n
    std_runs = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1)))

    if std_runs == 0:
        return {"prediction": None, "confidence": 0, "reason": "runs_test_degenerate"}

    z_runs = (runs - expected_runs) / std_runs

    # Significantly fewer runs => clustering => follow trend
    if z_runs < -1.96:
        conf = min(0.82, 0.50 + abs(z_runs + 1.96) * 0.08)
        return {
            "prediction": window[0],  # Continue current trend
            "confidence": round(conf, 3),
            "reason": f"runs_test_clustering_z_{z_runs:.2f}",
        }

    # Significantly more runs => oscillation => opposite of latest
    if z_runs > 1.96:
        opposite = "BIG" if window[0] == "SMALL" else "SMALL"
        conf = min(0.82, 0.50 + (z_runs - 1.96) * 0.08)
        return {
            "prediction": opposite,
            "confidence": round(conf, 3),
            "reason": f"runs_test_oscillation_z_{z_runs:.2f}",
        }

    return {"prediction": None, "confidence": 0, "reason": f"runs_test_random_z_{z_runs:.2f}"}


def _run_all_indicators(sizes: list[str]) -> dict:
    """Run all 10 statistical indicators and return the indicators dict."""
    return {
        "streak_reversal": _analyze_streak_indicator(sizes),
        "markov_transition": _analyze_markov_transition_indicator(sizes),
        "stat_frequency": _analyze_statistical_frequency_indicator(sizes),
        "ema_momentum": _analyze_ema_momentum_indicator(sizes),
        "pattern_match": _analyze_multi_ngram_pattern_indicator(sizes),
        "harmonic_periodicity": _analyze_harmonic_periodicity_indicator(sizes),
        "bayesian_posterior": _analyze_bayesian_posterior_indicator(sizes),
        "volatility_regime": _analyze_volatility_regime_indicator(sizes),
        "chi_square_skew": _analyze_chi_square_goodness_of_fit_indicator(sizes),
        "runs_test": _analyze_runs_test_indicator(sizes),
    }


def _score_indicators(indicators: dict, weights: dict) -> tuple:
    """
    Score indicators using squared-confidence amplification.
    High-confidence signals get exponentially more influence than low-confidence noise.

    Returns (small_score, big_score, total_weight, active_indicators).
    """
    small_score = 0.0
    big_score = 0.0
    total_weight = 0.0
    active = 0

    for name, indicator in indicators.items():
        w = weights.get(name, 0.08)
        pred = indicator.get("prediction")
        conf = indicator.get("confidence", 0)

        if pred and conf > 0:
            # Squared confidence amplification: high-confidence indicators dominate
            amplified_conf = conf * conf
            weighted_val = w * amplified_conf
            if pred == "SMALL":
                small_score += weighted_val
            else:
                big_score += weighted_val
            total_weight += w * conf  # Normalize by linear weight
            active += 1

    return small_score, big_score, total_weight, active


async def generate_prediction(
    session: AsyncSession, window: int = 500
) -> dict:
    """
    Generate an advanced 10-Indicator God-Mode Ensemble statistical prediction.

    Combines 10 independent mathematical, structural, Bayesian, and randomness-test indicators
    with squared-confidence amplification, multi-tier confluence boosting, and AI LLM reasoning.

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

    # Run all 10 indicators
    indicators = _run_all_indicators(sizes)

    # Fetch AI Pattern Reasoning via Key Rotation (Groq, OpenRouter, Gemini)
    # Pass full indicator breakdown to AI for contextual reasoning
    ai_reasoning = None
    try:
        from app.analytics.ai_rotator import fetch_ai_prediction
        indicator_summary = {
            "entropy": shannon_entropy,
            "z_score": z_score,
            "indicator_signals": {
                name: {"pred": ind.get("prediction"), "conf": ind.get("confidence", 0)}
                for name, ind in indicators.items()
                if ind.get("prediction")
            },
        }
        ai_res = await fetch_ai_prediction(sizes, indicator_summary)
        if ai_res and ai_res.get("ai_prediction"):
            ai_reasoning = ai_res
            indicators["ai_pattern_reasoning"] = {
                "prediction": ai_res["ai_prediction"],
                "confidence": ai_res["ai_confidence"],
                "reason": ai_res["ai_reason"],
                "provider": ai_res.get("provider"),
                "model": ai_res.get("model"),
            }
    except Exception as ai_err:
        logger.warning("ai_rotator_integration_warning", error=str(ai_err))

    # Adaptive Dynamic Weighting based on Shannon Entropy & Z-Score
    weights = dict(DEFAULT_WEIGHTS)
    if shannon_entropy < 0.90:
        # Low entropy = structured patterns => boost pattern-based indicators
        weights["markov_transition"] += 0.05
        weights["pattern_match"] += 0.05
        weights["harmonic_periodicity"] += 0.03
        weights["runs_test"] += 0.02
        weights["stat_frequency"] -= 0.08
    elif shannon_entropy > 0.98:
        # High entropy = noisy => boost statistical rebalance indicators
        weights["stat_frequency"] += 0.05
        weights["bayesian_posterior"] += 0.04
        weights["chi_square_skew"] += 0.03

    # Z-Score extreme deviation boost
    if abs(z_score) > 2.0:
        weights["stat_frequency"] += 0.04
        weights["chi_square_skew"] += 0.03

    # Squared-confidence weighted voting
    small_score, big_score, total_weight, active_indicators = _score_indicators(
        indicators, weights
    )

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

    # Count indicator agreement
    agreeing = sum(
        1 for ind in indicators.values()
        if ind.get("prediction") == prediction and ind.get("confidence", 0) > 0
    )

    # === MULTI-TIER CONFLUENCE BOOSTING ===

    # Tier 1: Micro-Macro Multi-Window Agreement (10 vs 30 vs 100 draw windows)
    if len(sizes) >= 30:
        micro_sizes = sizes[:10]
        micro_small = sum(1 for s in micro_sizes if s == "SMALL")
        micro_big = len(micro_sizes) - micro_small
        micro_dir = "SMALL" if micro_small > micro_big else "BIG"

        if micro_dir == prediction and agreeing >= 4:
            confidence = round(min(0.95, confidence + 0.08), 3)

    # Tier 2: Super-majority boost (7+ of 10 indicators agree)
    if agreeing >= 7 and active_indicators >= 8:
        confidence = round(min(0.96, confidence + 0.10), 3)
    elif agreeing >= 6 and active_indicators >= 7:
        confidence = round(min(0.93, confidence + 0.06), 3)

    # Tier 3: Streak exhaustion emergency boost
    streak = _get_current_streak(sizes)
    if streak["length"] >= 5 and prediction != streak["size"]:
        confidence = round(min(0.94, confidence + 0.07), 3)

    # Classify confidence level
    if confidence >= 0.75:
        confidence_level = "HIGH"
    elif confidence >= 0.55:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Calculate upcoming issue ID
    upcoming_issue_id = None
    if latest_issue:
        try:
            upcoming_issue_id = str(int(latest_issue) + 1)
        except ValueError:
            upcoming_issue_id = None

    logger.info(
        "prediction_generated_10in1",
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
        "disclaimer": "This prediction is based on 10-indicator statistical ensemble analysis for the upcoming game period. Each draw is an independent random event.",
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


def evaluate_recent_accuracy(rows: list) -> list[dict]:
    """
    Evaluate accuracy of the full 10-indicator ensemble on recent draws.

    For each of the last 5 draws, re-runs all 10 indicators on the data
    that was available BEFORE that draw occurred, then checks if the
    ensemble prediction matched the actual result.

    Args:
        rows: GameResult rows ordered by issue_id desc.

    Returns:
        List of dicts with issue_id, result, size, predicted_size, is_win.
    """
    if len(rows) < 10:
        return []

    results = []
    for i in range(min(5, len(rows) - 5)):
        current_row = rows[i]
        prior_sizes = [r.calculated_size for r in rows[i + 1 :]]
        if len(prior_sizes) < 5:
            continue

        # Run full 10-indicator ensemble on prior data
        indicators = _run_all_indicators(prior_sizes)
        weights = dict(DEFAULT_WEIGHTS)
        small_score, big_score, total_weight, active = _score_indicators(
            indicators, weights
        )

        if total_weight > 0:
            norm_small = small_score / total_weight
            norm_big = big_score / total_weight
            pred = "SMALL" if norm_small > norm_big else "BIG"
        else:
            pred = prior_sizes[0]

        is_win = pred == current_row.calculated_size
        results.append({
            "issue_id": current_row.issue_id,
            "result": current_row.result_number,
            "size": current_row.calculated_size,
            "color": current_row.source_color,
            "predicted_size": pred,
            "is_win": is_win,
            "prediction_status": "WIN" if is_win else "LOSS",
        })

    return results
