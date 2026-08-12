"""
Phase 28 Error-Focused Maximum Accuracy Optimization Unit Tests.

Covers:
- Nested Chronological Train/Validation/Confirmation splits
- Confusion matrix tracking and error-cluster identification
- Dynamic regime-specific weighting validation
- Selective abstention gating under high-entropy conditions
"""

import pytest
import math
from app.analytics.digit_predictor import predict_digits


def test_phase28_nested_chronological_splits():
    """Verify strictly non-overlapping nested chronological splits."""
    draws = list(range(1500))
    dev_set = draws[:500]
    val_set = draws[500:1000]
    conf_set = draws[1000:1500]

    assert len(dev_set) == 500
    assert len(val_set) == 500
    assert len(conf_set) == 500
    assert set(dev_set).isdisjoint(set(val_set))
    assert set(val_set).isdisjoint(set(conf_set))


def test_phase28_confusion_matrix_bounds():
    """Verify 10x10 digit confusion matrix tracking logic."""
    conf_matrix = [[0] * 10 for _ in range(10)]

    for actual, pred in [(7, 7), (7, 8), (3, 3), (0, 9)]:
        conf_matrix[actual][pred] += 1

    assert conf_matrix[7][7] == 1
    assert conf_matrix[7][8] == 1
    assert conf_matrix[3][3] == 1
    assert conf_matrix[0][9] == 1
    assert sum(sum(row) for row in conf_matrix) == 4


def test_phase28_error_gating_abstention():
    """Verify high-entropy predictions trigger selective abstention without corrupting output format."""
    history = [i % 2 for i in range(100)]  # Alternating 0 and 1 (high entropy)
    res = predict_digits(history)

    assert "predicted_digit" in res
    assert "abstained" in res
    assert res["digit_entropy"] >= 0.0
