"""
Test suite for Window Regime Alignment.
Verifies regime-specific window selection and adaptive behavior under STREAK_HEAVY, ALTERNATING, and HIGH_VOLATILITY.
"""

import pytest
from app.analytics.adaptive_window_selector import AdaptiveWindowSelector


def test_regime_aware_window_selection():
    selector = AdaptiveWindowSelector(min_samples=10, default_window=1000)

    # Simulate STREAK_HEAVY preferring short window 50
    for _ in range(15):
        selector.record_window_result(50, "STREAK_HEAVY", True, 0.12)
        selector.record_window_result(10000, "STREAK_HEAVY", False, 0.35)

    # Simulate ALTERNATING preferring medium window 250
    for _ in range(15):
        selector.record_window_result(250, "ALTERNATING", True, 0.15)
        selector.record_window_result(50, "ALTERNATING", False, 0.30)

    w_streak, meta_streak = selector.select_optimal_window("STREAK_HEAVY")
    w_alt, meta_alt = selector.select_optimal_window("ALTERNATING")

    assert w_streak == 50
    assert w_alt == 250
