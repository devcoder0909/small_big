"""
Test suite for Window Overfitting Prevention.
Verifies that a candidate window cannot become Champion based on tiny sample sizes (< 50).
"""

import pytest
from app.analytics.adaptive_window_selector import AdaptiveWindowSelector


def test_window_overfitting_protection():
    selector = AdaptiveWindowSelector(min_samples=50, default_window=1000)

    # Window 25 wins 3 games out of 3 (100% win rate but tiny sample)
    for _ in range(3):
        selector.record_window_result(25, "STREAK_HEAVY", True, 0.10)

    # Must refuse selection and fallback to STABLE_DEFAULT due to sample sufficiency gate
    w, meta = selector.select_optimal_window("STREAK_HEAVY")
    assert w == 1000
    assert meta["reason"] == "insufficient_samples_using_stable_default"
    assert meta["evaluated_samples"] == 3
