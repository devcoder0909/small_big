"""
Unit test suite for AdaptiveWindowSelector.
Verifies sample sufficiency thresholds, fallback rules, composite scoring, and OOS result recording.
"""

import pytest
from app.analytics.adaptive_window_selector import (
    AdaptiveWindowSelector,
    CANDIDATE_WINDOWS,
    STABLE_DEFAULT_WINDOW,
    MIN_EVALUATION_SAMPLES,
)


def test_adaptive_selector_initialization():
    selector = AdaptiveWindowSelector(min_samples=50, default_window=1000)
    metrics = selector.get_metrics()
    assert metrics["total_evaluated_samples"] == 0
    assert metrics["min_required_samples"] == 50


def test_adaptive_selector_insufficient_samples_fallback():
    selector = AdaptiveWindowSelector(min_samples=50, default_window=1000)

    # Record 10 samples (below 50 threshold)
    for _ in range(10):
        selector.record_window_result(100, "STREAK_HEAVY", True, 0.20)

    win, meta = selector.select_optimal_window("STREAK_HEAVY")
    assert win == 1000
    assert meta["reason"] == "insufficient_samples_using_stable_default"


def test_adaptive_selector_optimal_window_selection():
    selector = AdaptiveWindowSelector(min_samples=20, default_window=1000)

    # Train window 250 with high win rate on ALTERNATING regime
    for _ in range(30):
        selector.record_window_result(250, "ALTERNATING", True, 0.15)
        selector.record_window_result(500, "ALTERNATING", False, 0.35)

    win, meta = selector.select_optimal_window("ALTERNATING")
    assert win == 250
    assert meta["reason"] == "optimal_composite_score"
    assert "composite_scores" in meta
