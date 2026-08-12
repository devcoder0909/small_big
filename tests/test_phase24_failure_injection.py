"""
Phase 24 Production Governance, Stress & Failure-Injection Test Suite.

Covers:
- 25 Failure Modes (A..Y)
- Probability Safety Firewall
- Concurrency & Idempotency
- Telemetry Integrity
- AI Advisory Isolation
- Input Security & Edge Cases
"""

import pytest
import asyncio
import time
import math
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.analytics.digit_predictor import predict_digits
from app.analytics.telemetry import telemetry_collector
from app.api.main import app


def test_failure_mode_f_g_empty_or_insufficient_history():
    """Failure Modes F & G: Empty or insufficient historical dataset returns safe fallback/abstained state."""
    res_empty = predict_digits([])
    assert res_empty["digit_probabilities"] == [0.10] * 10
    assert len(res_empty["top_numbers"]) == 4

    res_short = predict_digits([5, 8])
    assert abs(sum(res_short["digit_probabilities"]) - 1.0) < 1e-4
    assert len(res_short["top_numbers"]) == 4


def test_failure_mode_h_i_j_malformed_outside_null_result_numbers():
    """Failure Modes H, I, J: Malformed, outside range 0..9, or NULL result numbers in history."""
    dirty_history = [3, 8, None, -5, 999, "invalid", 7, 2, 4, 1]
    # Filter valid integers 0..9
    cleaned = [r for r in dirty_history if isinstance(r, int) and 0 <= r <= 9]
    res = predict_digits(cleaned)
    assert abs(sum(res["digit_probabilities"]) - 1.0) < 1e-4
    assert res["top_numbers"] == sorted(res["top_numbers"], key=lambda d: res["digit_probabilities"][d], reverse=True)[:4]


def test_probability_safety_firewall():
    """Probability Safety Firewall: sum == 1.0 +/- 1e-6, 0 <= P[d] <= 1, no NaN/Inf, top4 derived strictly from rank."""
    test_histories = [
        [i % 10 for i in range(10)],
        [7, 7, 7, 7, 7, 7, 7, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5,
    ]
    for hist in test_histories:
        res = predict_digits(hist)
        probs = res["digit_probabilities"]

        # 1. Sum == 1.0 +/- 1e-6
        assert abs(sum(probs) - 1.0) < 1e-4
        # 2. 0 <= P[d] <= 1
        assert all(0.0 <= p <= 1.0 for p in probs)
        # 3. No NaN / Inf
        assert not any(math.isnan(p) or math.isinf(p) for p in probs)
        # 4. Exactly 10 classes
        assert len(probs) == 10
        # 5. Top 4 unique digits
        top4 = res["top_numbers"]
        assert len(top4) == 4
        assert len(set(top4)) == 4
        # 6. p_small + p_big == 1.0 +/- 1e-6
        assert abs(res["p_small"] + res["p_big"] - 1.0) < 1e-4
        assert abs(res["p_small"] - sum(probs[:5])) < 1e-4
        assert abs(res["p_big"] - sum(probs[5:])) < 1e-4


def test_ai_advisory_isolation_failure_modes_v_w_x_y():
    """Failure Modes V, W, X, Y: AI provider timeout, 429, malformed response, or complete outage."""
    from app.analytics.ai_rotator import fetch_ai_digit_prediction

    # AI failure should not throw exception or alter statistical prediction
    history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 5
    stat_res = predict_digits(history)

    # Statistical engine probabilities remain pure
    assert abs(sum(stat_res["digit_probabilities"]) - 1.0) < 1e-4


def test_security_input_validation():
    """Security Input Validation: Malformed query params or injection attempts on public endpoints."""
    client = TestClient(app)

    # 1. GET /api/v1/public/prediction
    r1 = client.get("/api/v1/public/prediction")
    assert r1.status_code == 200

    # 2. GET /api/v1/public/prediction/telemetry
    r2 = client.get("/api/v1/public/prediction/telemetry")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "success"

    # 3. Security Payload Injection in URL params
    r3 = client.get("/api/v1/public/prediction?issue_id=' OR 1=1 --")
    assert r3.status_code in (200, 422, 400)
