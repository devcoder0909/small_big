"""Unit tests for the upgraded Prediction Engine."""

import pytest
from datetime import datetime, timezone
from app.analytics.prediction_engine import (
    generate_prediction,
    _calculate_shannon_entropy,
    _calculate_z_score,
    _analyze_streak_indicator,
    _analyze_markov_transition_indicator,
    _analyze_statistical_frequency_indicator,
    _analyze_ema_momentum_indicator,
    _analyze_multi_ngram_pattern_indicator,
)
from app.models.game_result import GameResult


def test_shannon_entropy_calculation():
    """Test Shannon entropy calculation on uniform vs skewed vs single-class sequences."""
    # Equal distribution -> Maximum entropy (1.0)
    sizes_equal = ["SMALL"] * 50 + ["BIG"] * 50
    assert _calculate_shannon_entropy(sizes_equal) == 1.0

    # Skewed distribution -> Lower entropy (< 1.0)
    sizes_skewed = ["SMALL"] * 80 + ["BIG"] * 20
    entropy = _calculate_shannon_entropy(sizes_skewed)
    assert 0 < entropy < 1.0

    # Pure single class -> Zero entropy
    sizes_pure = ["SMALL"] * 50
    assert _calculate_shannon_entropy(sizes_pure) == 0.0


def test_z_score_calculation():
    """Test Z-score calculation for frequency deviation."""
    sizes_balanced = ["SMALL", "BIG"] * 25
    z_bal, p_bal = _calculate_z_score(sizes_balanced)
    assert z_bal == 0.0
    assert p_bal == 0.5

    sizes_small_heavy = ["SMALL"] * 40 + ["BIG"] * 10
    z_heavy, p_heavy = _calculate_z_score(sizes_small_heavy)
    assert z_heavy > 2.0  # Statistically significant deviation
    assert p_heavy == 0.8


def test_streak_indicator():
    """Test streak indicator prediction on long streak."""
    # Current streak of 6 SMALLs
    sizes = ["SMALL"] * 6 + ["BIG", "BIG", "SMALL", "BIG", "SMALL", "BIG"]
    res = _analyze_streak_indicator(sizes)
    assert res["prediction"] in ("BIG", "SMALL")
    assert res["confidence"] > 0


def test_markov_transition_indicator():
    """Test Markov transition indicator on repeating sequence."""
    sizes = ["BIG", "SMALL", "BIG", "SMALL", "BIG", "SMALL"] * 5
    res = _analyze_markov_transition_indicator(sizes)
    assert res["prediction"] is not None
    assert res["confidence"] > 0


def test_ema_momentum_indicator():
    """Test Dual EMA momentum indicator."""
    sizes = ["SMALL"] * 15 + ["BIG"] * 15
    res = _analyze_ema_momentum_indicator(sizes)
    assert res["prediction"] is not None


def test_multi_ngram_pattern_indicator():
    """Test N-gram pattern indicator on pattern repetition."""
    # Repeating sequence of SMALL, BIG, SMALL, BIG...
    sizes = ["SMALL", "BIG"] * 30
    res = _analyze_multi_ngram_pattern_indicator(sizes)
    assert res["prediction"] is not None
    assert res["confidence"] > 0


@pytest.mark.asyncio
async def test_end_to_end_prediction_generation(db_session):
    """Test end-to-end prediction generation on populated mock database."""
    now = datetime.now(timezone.utc)
    mock_sizes = ["SMALL", "BIG", "SMALL", "SMALL", "BIG", "BIG", "SMALL", "BIG"] * 10

    for i, size in enumerate(mock_sizes):
        db_session.add(
            GameResult(
                issue_id=str(1000 + i),
                result_number=3 if size == "SMALL" else 8,
                source_color="green",
                calculated_size=size,
                first_observed_at=now,
                last_observed_at=now,
                source_url="http://test",
            )
        )
    await db_session.commit()

    prediction = await generate_prediction(db_session)
    assert prediction["status"] == "ACTIVE"
    assert prediction["prediction"] in ("SMALL", "BIG")
    assert prediction["confidence"] >= 0.40
    assert "shannon_entropy" in prediction
    assert "z_score" in prediction
    assert prediction["upcoming_issue_id"] == str(1000 + len(mock_sizes))
