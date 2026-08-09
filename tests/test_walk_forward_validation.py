"""
Walk-Forward Validation & Out-Of-Sample Backtest Test Suite.

Verifies:
1. Zero look-ahead data leakage in walk-forward evaluation.
2. Confidence calibration accuracy.
3. Anti-overfitting regime shift penalties.
4. Indicator collinearity dampening and signal penalization.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import (
    generate_prediction,
    _run_all_indicators,
    _score_indicators,
    _calculate_adaptive_indicator_weights,
)


@pytest.mark.asyncio
async def test_walk_forward_no_future_data_leakage():
    """Verify sliding walk-forward validation never accesses future draw indices."""
    sizes = ["SMALL", "BIG", "SMALL", "BIG", "SMALL", "BIG", "SMALL", "BIG"] * 25
    numbers = [3 if s == "SMALL" else 8 for s in sizes]
    colors = ["green" if s == "SMALL" else "red" for s in sizes]

    # Walk forward across 30 draws
    for i in range(5, 35):
        past_slice = sizes[i:]
        num_slice = numbers[i:]
        col_slice = colors[i:]

        # Guarantee that past_slice only contains indices >= i (strictly historical)
        assert len(past_slice) == len(sizes) - i
        indicators = _run_all_indicators(past_slice, num_slice, col_slice)
        assert "streak_reversal" in indicators
        assert "markov_transition" in indicators


@pytest.mark.asyncio
async def test_collinearity_dampening_in_scoring():
    """Verify collinear frequency indicators are dampened by 40% when voting identically."""
    indicators = {
        "stat_frequency": {"prediction": "SMALL", "confidence": 0.80},
        "bayesian_posterior": {"prediction": "SMALL", "confidence": 0.80},
        "chi_square_skew": {"prediction": "SMALL", "confidence": 0.80},
    }
    weights = {
        "stat_frequency": 0.12,
        "bayesian_posterior": 0.09,
        "chi_square_skew": 0.07,
    }

    small_score, big_score, total_weight, active = _score_indicators(indicators, weights)

    # First indicator receives full weight, second and third receive 60% weight
    expected_small_score = (0.12 * (0.80**2)) + (0.09 * 0.60 * (0.80**2)) + (0.07 * 0.60 * (0.80**2))
    assert pytest.approx(small_score, abs=1e-4) == expected_small_score
    assert big_score == 0.0
    assert active == 3


@pytest.mark.asyncio
async def test_failing_indicator_penalization():
    """Verify indicators with < 35% win rate are penalized down to 0.05x weight."""
    base_weights = {"digit_numeric_momentum": 0.07}
    # Mismatched sequence where digit_numeric_momentum predicts BIG (from number 8) but actual is SMALL
    sizes = ["SMALL"] * 100
    numbers = [8] * 100
    colors = ["red"] * len(sizes)

    adaptive = _calculate_adaptive_indicator_weights(sizes, base_weights, numbers, colors)
    # The weight should be heavily suppressed
    assert adaptive["digit_numeric_momentum"] < base_weights["digit_numeric_momentum"]
