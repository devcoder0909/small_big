"""
V3 Regime Detector — Market Regime Classification Engine.

Classifies historical draw sequence into one of 5 distinct market regimes
using ONLY historical data available prior to the target period:

1. STREAK_HEAVY — Long consecutive streak of BIG or SMALL (streak >= 4).
2. ALTERNATING — Rapid 1-1-1 oscillation pattern (e.g. B-S-B-S-B-S).
3. FREQUENCY_IMBALANCED — Significant Z-score frequency deviation (|Z| >= 1.8).
4. HIGH_VOLATILITY — High Shannon entropy (entropy >= 0.98) & regime shifts.
5. STABLE_NEUTRAL — Standard balanced statistical random walk.
"""

import math


def _compute_z_score(sizes: list[str]) -> float:
    """Compute Z-score frequency deviation for historical sizes."""
    if len(sizes) < 10:
        return 0.0
    n = len(sizes)
    small_count = sum(1 for s in sizes if s == "SMALL")
    p = 0.5
    mean = n * p
    std_dev = math.sqrt(n * p * (1 - p))
    if std_dev == 0:
        return 0.0
    return round((small_count - mean) / std_dev, 4)


def _get_current_streak_len(sizes: list[str]) -> int:
    """Calculate length of active streak."""
    if not sizes:
        return 0
    curr = sizes[0]
    count = 0
    for s in sizes:
        if s == curr:
            count += 1
        else:
            break
    return count


def detect_market_regime(sizes: list[str], shannon_entropy: float = 0.95) -> dict:
    """
    Classify the current market regime from historical sizes (newest first).

    Args:
        sizes: List of "BIG"/"SMALL" strings (newest first).
        shannon_entropy: Shannon entropy of recent 50 draws.

    Returns:
        Dict with regime name, description, and regime parameters.
    """
    if len(sizes) < 10:
        return {
            "regime": "STABLE_NEUTRAL",
            "description": "Insufficient history for regime classification",
            "streak_length": 1,
            "z_score": 0.0,
            "entropy": shannon_entropy,
        }

    curr_streak = _get_current_streak_len(sizes)
    z_val = _compute_z_score(sizes[:100] if len(sizes) >= 100 else sizes)

    # 1. Streak-heavy regime check
    if curr_streak >= 4:
        return {
            "regime": "STREAK_HEAVY",
            "description": f"Active streak of {curr_streak} consecutive {sizes[0]} draws",
            "streak_length": curr_streak,
            "z_score": z_val,
            "entropy": shannon_entropy,
        }

    # 2. Alternating regime check (last 6 draws alternating)
    if len(sizes) >= 6:
        is_alternating = all(sizes[i] != sizes[i + 1] for i in range(5))
        if is_alternating:
            return {
                "regime": "ALTERNATING",
                "description": "Rapid 1-1-1 oscillation pattern active",
                "streak_length": 1,
                "z_score": z_val,
                "entropy": shannon_entropy,
            }

    # 3. Frequency imbalanced check (|Z| >= 1.8)
    if abs(z_val) >= 1.8:
        target_side = "SMALL" if z_val > 0 else "BIG"
        return {
            "regime": "FREQUENCY_IMBALANCED",
            "description": f"Significant statistical frequency deviation (|Z|={abs(z_val):.2f}) favoring {target_side}",
            "streak_length": curr_streak,
            "z_score": z_val,
            "entropy": shannon_entropy,
        }

    # 4. High volatility check
    if shannon_entropy >= 0.98:
        return {
            "regime": "HIGH_VOLATILITY",
            "description": f"High sequence Shannon entropy ({shannon_entropy:.4f})",
            "streak_length": curr_streak,
            "z_score": z_val,
            "entropy": shannon_entropy,
        }

    # 5. Default stable neutral
    return {
        "regime": "STABLE_NEUTRAL",
        "description": "Standard balanced statistical random walk",
        "streak_length": curr_streak,
        "z_score": z_val,
        "entropy": shannon_entropy,
    }
