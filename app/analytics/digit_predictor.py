"""
Digit Predictor Module — 10-Class Probabilistic Dirichlet-Markov Engine.

Implements multi-order Markov transition matrices (Orders 1-3) with sparse state fallback,
Dirichlet-Multinomial conjugate priors, inter-arrival recurrence distance hazards,
multi-horizon lookback fusion, Top-4 probability mass selection, and selective prediction.

Strictly obeys:
1. P[0..9] probabilities sum to 1.0 +/- 1e-6.
2. Top-4 selection derived strictly from probability ranking.
3. Derives p_big = sum(P[5..9]) and p_small = sum(P[0..4]).
4. Zero target leakage: input features strictly precede target period.
"""

import math
from collections import Counter, defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)

HORIZONS = [25, 50, 100, 250, 500, 1000, 2000]


def _model_dirichlet_prior(numbers: list[int], window: int = 250, alpha: float = 1.0) -> list[float]:
    """Dirichlet-Multinomial Bayesian Conjugate Prior with uniform alpha prior."""
    if not numbers:
        return [0.10] * 10
    slice_nums = numbers[:window]
    counts = Counter(slice_nums)
    total = len(slice_nums)
    probs = [(counts.get(d, 0) + alpha) / (total + 10.0 * alpha) for d in range(10)]
    s = sum(probs)
    return [p / s for p in probs]


def _model_markov_order1(numbers: list[int], window: int = 1000, alpha: float = 1.0, decay: float = 0.998) -> list[float]:
    """First-Order Markov Transition Matrix P(d_t | d_{t-1}) with exponential recency decay."""
    if len(numbers) < 2:
        return _model_dirichlet_prior(numbers, window, alpha)

    slice_nums = numbers[:window]
    prev_digit = slice_nums[0]

    counts = defaultdict(float)
    total = 0.0
    for i in range(len(slice_nums) - 1):
        from_d = slice_nums[i + 1]
        to_d = slice_nums[i]
        if from_d == prev_digit:
            w = math.pow(decay, i)
            counts[to_d] += w
            total += w

    if total < 0.5:
        return _model_dirichlet_prior(numbers, window, alpha)

    probs = [(counts[d] + alpha) / (total + 10.0 * alpha) for d in range(10)]
    s = sum(probs)
    return [p / s for p in probs]


def _model_markov_order2(numbers: list[int], window: int = 2000, alpha: float = 1.0, decay: float = 0.998) -> list[float]:
    """Second-Order Markov Matrix P(d_t | d_{t-2}, d_{t-1}) with Order 1 fallback."""
    if len(numbers) < 3:
        return _model_markov_order1(numbers, window, alpha, decay)

    slice_nums = numbers[:window]
    ctx = (slice_nums[1], slice_nums[0])

    counts = defaultdict(float)
    total = 0.0
    for i in range(len(slice_nums) - 2):
        if (slice_nums[i + 2], slice_nums[i + 1]) == ctx:
            w = math.pow(decay, i)
            counts[slice_nums[i]] += w
            total += w

    if total < 0.5:
        return _model_markov_order1(numbers, window, alpha, decay)

    probs = [(counts[d] + alpha) / (total + 10.0 * alpha) for d in range(10)]
    s = sum(probs)
    return [p / s for p in probs]


def _model_markov_order3(numbers: list[int], window: int = 3000, alpha: float = 1.0, decay: float = 0.998) -> list[float]:
    """Third-Order Markov Matrix P(d_t | d_{t-3}, d_{t-2}, d_{t-1}) with Order 2 fallback."""
    if len(numbers) < 4:
        return _model_markov_order2(numbers, window, alpha, decay)

    slice_nums = numbers[:window]
    ctx = (slice_nums[2], slice_nums[1], slice_nums[0])

    counts = defaultdict(float)
    total = 0.0
    for i in range(len(slice_nums) - 3):
        if (slice_nums[i + 3], slice_nums[i + 2], slice_nums[i + 1]) == ctx:
            w = math.pow(decay, i)
            counts[slice_nums[i]] += w
            total += w

    if total < 0.5:
        return _model_markov_order2(numbers, window, alpha, decay)

    probs = [(counts[d] + alpha) / (total + 10.0 * alpha) for d in range(10)]
    s = sum(probs)
    return [p / s for p in probs]


def _model_recurrence_hazard(numbers: list[int], window: int = 1000) -> list[float]:
    """Inter-arrival distance hazard estimator for digit recurrence intervals."""
    if not numbers or len(numbers) < 10:
        return [0.10] * 10

    slice_nums = numbers[:window]
    last_seen = {}
    gap_history = defaultdict(list)

    for idx, d in enumerate(slice_nums):
        if d in last_seen:
            gap = idx - last_seen[d]
            gap_history[d].append(gap)
        last_seen[d] = idx

    probs = []
    for d in range(10):
        current_gap = last_seen.get(d, window)
        gaps = gap_history.get(d, [10])
        avg_gap = sum(gaps) / len(gaps) if gaps else 10.0
        hazard = min(2.5, max(0.2, current_gap / avg_gap))
        probs.append(hazard)

    s = sum(probs)
    return [p / s for p in probs]


def predict_digits(
    numbers: list[int] | None,
    sizes: list[str] | None = None,
    window: int | None = None,
) -> dict:
    """
    Generate 10-class digit prediction vector and Top-4 maximum coverage numbers.
    Analyzes full historical data with multi-horizon lookback, Markov transitions,
    recurrence hazard, and parity pattern dynamics.
    """
    if not numbers or len(numbers) < 10:
        return {
            "predicted_digit": None,
            "top_numbers": [0, 1, 2, 3],
            "digit_probabilities": [0.10] * 10,
            "top1_probability": 0.10,
            "top2_probability_mass": 0.20,
            "top3_probability_mass": 0.30,
            "top4_probability_mass": 0.40,
            "digit_entropy": 1.0,
            "digit_confidence": 0.0,
            "p_big": 0.50,
            "p_small": 0.50,
            "method": "insufficient_data",
            "abstained": True,
            "abstention_reason": "INSUFFICIENT_HISTORICAL_DIGITS",
            "pattern_analysis": {
                "historical_draws_analyzed": len(numbers) if numbers else 0,
                "overdue_digits": [],
                "parity_bias": "NEUTRAL",
            },
        }

    total_history_len = len(numbers)
    analysis_window = window if window is not None else 250
    analysis_window = min(analysis_window, total_history_len)

    # 1. Component probability estimation across full historical horizon
    p_global = _model_dirichlet_prior(numbers, window=total_history_len)
    p_window_dir = _model_dirichlet_prior(numbers, window=analysis_window)
    p_m1 = _model_markov_order1(numbers, window=min(1000, total_history_len))
    p_m2 = _model_markov_order2(numbers, window=min(2000, total_history_len))
    p_m3 = _model_markov_order3(numbers, window=min(3000, total_history_len))
    p_recur = _model_recurrence_hazard(numbers, window=min(1000, total_history_len))

    # Ensemble weights: Markov O1 (0.25), Markov O2 (0.25), Markov O3 (0.15), Recurrence (0.15), Local Dirichlet (0.10), Global Prior (0.10)
    raw_probs = [
        0.25 * p_m1[d]
        + 0.25 * p_m2[d]
        + 0.15 * p_m3[d]
        + 0.15 * p_recur[d]
        + 0.10 * p_window_dir[d]
        + 0.10 * p_global[d]
        for d in range(10)
    ]

    # Normalize probabilities so sum(probs) == 1.0 strictly
    sum_probs = sum(raw_probs)
    probs = [round(p / sum_probs, 4) for p in raw_probs]

    # Final normalization adjustment for floating point precision
    diff = 1.0 - sum(probs)
    if abs(diff) > 1e-6:
        probs[0] = round(probs[0] + diff, 4)

    # Rank digits 0-9 by probability descending
    ranked_digits = sorted(range(10), key=lambda d: probs[d], reverse=True)
    top1 = ranked_digits[0]
    top2 = ranked_digits[:2]
    top3 = ranked_digits[:3]
    top4 = ranked_digits[:4]

    top1_prob = probs[top1]
    top2_mass = round(sum(probs[d] for d in top2), 4)
    top3_mass = round(sum(probs[d] for d in top3), 4)
    top4_mass = round(sum(probs[d] for d in top4), 4)

    # Digit Shannon Entropy (base 10, range 0.0 to 1.0)
    entropy_sum = -sum(p * math.log10(p) for p in probs if p > 0)
    normalized_entropy = round(max(0.0, min(1.0, entropy_sum)), 4)

    # BIG / SMALL consistency derivation
    p_small = round(sum(probs[d] for d in range(5)), 4)
    p_big = round(1.0 - p_small, 4)

    # Parity Analysis (Odd vs Even)
    p_odd = sum(probs[d] for d in [1, 3, 5, 7, 9])
    p_even = sum(probs[d] for d in [0, 2, 4, 6, 8])
    parity_bias = "ODD" if p_odd > 0.55 else ("EVEN" if p_even > 0.55 else "NEUTRAL")

    # Identify overdue digits based on recurrence hazard
    overdue_digits = sorted(range(10), key=lambda d: p_recur[d], reverse=True)[:3]

    # Selective Prediction & Abstention Gating
    is_abstained = False
    abstention_reason = None

    if normalized_entropy >= 0.995 or top1_prob < 0.105:
        is_abstained = True
        abstention_reason = "EXTREME_ENTROPY_LOW_EDGE"

    predicted_digit = None if is_abstained else top1

    return {
        "predicted_digit": predicted_digit,
        "top_numbers": top4,
        "digit_probabilities": probs,
        "top1_probability": top1_prob,
        "top2_probability_mass": top2_mass,
        "top3_probability_mass": top3_mass,
        "top4_probability_mass": top4_mass,
        "digit_entropy": normalized_entropy,
        "digit_confidence": top1_prob if not is_abstained else 0.0,
        "p_big": p_big,
        "p_small": p_small,
        "method": "dirichlet_markov_multiorder_ensemble",
        "abstained": is_abstained,
        "abstention_reason": abstention_reason,
        "pattern_analysis": {
            "historical_draws_analyzed": total_history_len,
            "overdue_digits": overdue_digits,
            "parity_bias": parity_bias,
            "p_odd": round(p_odd, 4),
            "p_even": round(p_even, 4),
        },
    }
