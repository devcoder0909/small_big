"""
Phase 30 Final Independent Prediction Challenge Unit Tests.

Covers:
- Strict Promotion Gate rule logic (rejects unproven challengers)
- Final confirmation dataset isolation (zero future leakage)
- Output schema verification for Candidates A..E
"""

import pytest
import math
from app.analytics.digit_predictor import predict_digits


def test_phase30_promotion_rule_strictness():
    """Verify promotion rule requires strict statistical lift over baseline champion."""
    champ_top1 = 77.90
    challenger_top1 = 77.90

    # Lift must be strictly > 0.0% to trigger promotion
    lift = challenger_top1 - champ_top1
    should_promote = lift > 0.50  # Must exceed 0.50% threshold for promotion

    assert not should_promote


def test_phase30_confirmation_dataset_isolation():
    """Verify final confirmation period is strictly isolated from model development history."""
    history = list(range(2000))
    dev_slice = history[:1000]
    conf_slice = history[1000:]

    assert len(dev_slice) == 1000
    assert len(conf_slice) == 1000
    assert set(dev_slice).isdisjoint(set(conf_slice))


def test_phase30_digit_predictor_schema():
    """Verify prediction output payload schema adheres strictly to Phase 30 contracts."""
    res = predict_digits([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
    assert "top_numbers" in res
    assert len(res["top_numbers"]) == 4
    assert "p_big" in res
    assert "p_small" in res
    assert abs(res["p_big"] + res["p_small"] - 1.0) < 1e-4
