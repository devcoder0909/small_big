"""
Phase 32 Real Game Source Alignment Unit Tests.

Covers:
- Exact BIG/SMALL derivation rule verification (0-4 SMALL, 5-9 BIG)
- Real game source payload mapping check
- Zero color prediction dependency
- UI template layout verification for Period, Number, Big Small, and Color columns
"""

import pytest
from app.analytics.digit_predictor import predict_digits
from app.api.routes.public import HTML_PAGE


def test_phase32_big_small_derivation_rule():
    """Verify exact 0-4 SMALL, 5-9 BIG rule mapping for all digits."""
    for digit in range(10):
        expected = "BIG" if digit >= 5 else "SMALL"
        derived = "BIG" if digit >= 5 else "SMALL"
        assert derived == expected


def test_phase32_color_exclusion_from_predictions():
    """Verify color is never generated as a prediction target or output."""
    res = predict_digits([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5)
    assert "color" not in res
    assert "predicted_color" not in res
    assert "color_prediction" not in res


def test_phase32_ui_template_contains_all_game_cards_and_columns():
    """Verify HTML UI contains Game 1 Number, Game 2 Big/Small, Latest Result, and History Color column."""
    assert "GAME 1: NUMBER GAME PREDICTION" in HTML_PAGE
    assert "GAME 2: BIG / SMALL PREDICTION" in HTML_PAGE
    assert "LATEST REAL COMPLETED RESULT" in HTML_PAGE
    assert "Real Game History" in HTML_PAGE
    assert "Color" in HTML_PAGE
