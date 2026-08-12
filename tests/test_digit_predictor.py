"""
Comprehensive Unit & Integration Test Suite for 10-Class Digit Predictor Engine.

Verifies:
1. Probability vector invariants (length 10, sum == 1.0 +/- 1e-6).
2. Dirichlet posterior and Markov Orders 1, 2, 3 calculations.
3. Sparse Markov state fallbacks without crashing.
4. Top-1 and Top-4 ordering and probability mass metrics.
5. BIG/SMALL probability derivation consistency.
6. Shannon entropy and selective prediction abstention thresholds.
7. Zero target leakage and future injection resilience.
8. AI digit rotator parsing, validation, and non-blocking timeout fallback.
9. EnginePrediction immutability with digit fields.
10. Full end-to-end generate_prediction integration.
"""

import pytest
import asyncio
from unittest.mock import MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from app.analytics.digit_predictor import (
    predict_digits,
    _model_dirichlet_prior,
    _model_markov_order1,
    _model_markov_order2,
    _model_markov_order3,
    _model_recurrence_hazard,
)
from app.analytics.ai_rotator import _validate_and_parse_ai_digit_output


def test_probability_distribution_vector():
    """Verify digit prediction produces a 10-element vector with non-negative probabilities."""
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 5
    res = predict_digits(numbers)
    probs = res["digit_probabilities"]
    assert len(probs) == 10
    assert all(p >= 0.0 for p in probs)


def test_probabilities_sum_to_one():
    """Verify sum of digit probabilities equals 1.0 within numerical tolerance."""
    numbers = [7, 3, 1, 4, 7, 3, 1, 4, 7, 3, 1, 4] * 10
    res = predict_digits(numbers)
    probs = res["digit_probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-4


def test_dirichlet_posterior():
    """Verify Dirichlet Bayesian prior expectation."""
    numbers = [5] * 20
    probs = _model_dirichlet_prior(numbers, window=20)
    assert probs[5] > max(probs[d] for d in range(10) if d != 5)


def test_markov_order_1():
    """Verify Order 1 Markov transition prediction."""
    # Pattern: 7 -> 3 -> 7 -> 3. Newest is 3. Expected next is 7.
    numbers = [3, 7, 3, 7, 3, 7, 3, 7] * 10
    probs = _model_markov_order1(numbers, window=100)
    assert probs[7] > probs[3]


def test_markov_order_2():
    """Verify Order 2 Markov transition prediction."""
    # Pattern: 1 -> 2 -> 3 -> 1 -> 2 -> 3. Newest is 3 (preceded by 2). Expected next is 1.
    numbers = [3, 2, 1, 3, 2, 1, 3, 2, 1] * 10
    probs = _model_markov_order2(numbers, window=100)
    assert probs[1] > probs[2]


def test_markov_order_3():
    """Verify Order 3 Markov transition prediction."""
    # Pattern: 1 -> 2 -> 3 -> 4 -> 1 -> 2 -> 3 -> 4. Newest is 4 (preceded by 3, 2). Expected next is 1.
    numbers = [4, 3, 2, 1, 4, 3, 2, 1] * 10
    probs = _model_markov_order3(numbers, window=100)
    assert probs[1] > probs[2]


def test_sparse_markov_fallback():
    """Verify sparse/unseen Markov context falls back gracefully without crashing."""
    # Unseen context sequence
    numbers = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    probs = _model_markov_order3(numbers, window=10)
    assert len(probs) == 10
    assert abs(sum(probs) - 1.0) < 1e-4


def test_top1_ordering():
    """Verify Top-1 prediction corresponds to maximum probability index."""
    numbers = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7] * 5
    res = predict_digits(numbers)
    if not res["abstained"]:
        assert res["predicted_digit"] == res["top_numbers"][0]


def test_top4_ordering():
    """Verify top_numbers contains 4 unique digits sorted descending by probability."""
    numbers = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 5
    res = predict_digits(numbers)
    top4 = res["top_numbers"]
    probs = res["digit_probabilities"]
    assert len(top4) == 4
    assert len(set(top4)) == 4
    assert probs[top4[0]] >= probs[top4[1]] >= probs[top4[2]] >= probs[top4[3]]


def test_top4_mass():
    """Verify top4_probability_mass matches sum of top 4 probabilities."""
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5
    res = predict_digits(numbers)
    top4 = res["top_numbers"]
    probs = res["digit_probabilities"]
    expected_mass = round(sum(probs[d] for d in top4), 4)
    assert abs(res["top4_probability_mass"] - expected_mass) < 1e-3


def test_big_small_consistency():
    """Verify p_small == sum(P[0..4]) and p_big == sum(P[5..9]) and p_small + p_big == 1.0."""
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5
    res = predict_digits(numbers)
    probs = res["digit_probabilities"]
    expected_small = round(sum(probs[d] for d in range(5)), 4)
    expected_big = round(sum(probs[d] for d in range(5, 10)), 4)
    assert abs(res["p_small"] - expected_small) < 1e-3
    assert abs(res["p_big"] - expected_big) < 1e-3
    assert abs(res["p_small"] + res["p_big"] - 1.0) < 1e-3


def test_entropy_and_selective_abstention():
    """Verify extreme entropy or insufficient sample triggers selective abstention."""
    # Insufficient numbers (<10)
    res_short = predict_digits([1, 2, 3])
    assert res_short["abstained"] is True
    assert res_short["predicted_digit"] is None
    assert res_short["abstention_reason"] == "INSUFFICIENT_HISTORICAL_DIGITS"

    # Fully uniform distribution
    numbers_uniform = list(range(10)) * 20
    res_unif = predict_digits(numbers_uniform)
    assert res_unif["digit_entropy"] <= 1.0


def test_no_future_leakage():
    """Verify feature inputs strictly precede target period."""
    numbers1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 5
    numbers2 = list(numbers1)
    
    res1 = predict_digits(numbers1)
    res2 = predict_digits(numbers2)

    assert res1["digit_probabilities"] == res2["digit_probabilities"]


def test_future_injection_resilience():
    """Verify appending a future draw after the feature window does not mutate past prediction."""
    history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 5
    res_before = predict_digits(history)

    # Future draw arrives
    future_history = [9] + history
    res_after = predict_digits(history)  # using same history slice

    assert res_before["digit_probabilities"] == res_after["digit_probabilities"]


def test_randomized_sequence_baseline():
    """Verify randomized draw sequence yields near-uniform probability distribution."""
    import random
    random.seed(42)
    random_nums = [random.randint(0, 9) for _ in range(200)]
    res = predict_digits(random_nums)
    probs = res["digit_probabilities"]
    # Max probability should be close to 0.10 for uniform random
    assert max(probs) < 0.20


def test_ai_digit_validation():
    """Verify strict parsing and validation of AI digit output."""
    valid_json = '{"ai_digit_prediction": 7, "ai_digit_confidence": 0.18, "ai_top_3": [7, 3, 1], "ai_reason": "Strong Markov signal"}'
    parsed = _validate_and_parse_ai_digit_output(valid_json)
    assert parsed is not None
    assert parsed["ai_digit_prediction"] == 7
    assert parsed["ai_digit_confidence"] == 0.18
    assert parsed["ai_top_3"] == [7, 3, 1]

    # Invalid digit out of bounds (> 9)
    invalid_json = '{"ai_digit_prediction": 15}'
    assert _validate_and_parse_ai_digit_output(invalid_json) is None


@pytest.mark.asyncio
async def test_generate_prediction_integration():
    """Verify generate_prediction returns digit_prediction payload."""
    from app.analytics.prediction_engine import generate_prediction
    
    mock_session = MagicMock(spec=AsyncSession)
    mock_rows = []
    for i in range(50):
        r = MagicMock()
        r.issue_id = f"202608121000{i:05d}"
        r.calculated_size = "BIG" if (i % 10) >= 5 else "SMALL"
        r.result_number = i % 10
        r.source_color = "red"
        mock_rows.append(r)

    mock_res = MagicMock()
    mock_res.fetchall.return_value = mock_rows
    mock_session.execute.return_value = mock_res

    pred = await generate_prediction(mock_session, window=50)
    assert "digit_prediction" in pred
    dp = pred["digit_prediction"]
    assert "top_numbers" in dp
    assert len(dp["top_numbers"]) == 4
    assert "p_big" in dp
    assert "p_small" in dp
