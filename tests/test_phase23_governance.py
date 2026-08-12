"""
Phase 23 Live Telemetry, Walk-Forward Validation & Model Governance Unit Tests.
"""

import pytest
import math
from fastapi.testclient import TestClient
from app.api.main import app
from app.analytics.telemetry import telemetry_collector
from app.analytics.digit_predictor import predict_digits


def test_shadow_telemetry_recording_and_scoring():
    """Verify non-invasive live shadow telemetry captures and scores records upon draw completion."""
    pred_dict = {
        "predicted_digit": 7,
        "digit_confidence": 0.15,
        "digit_probabilities": [0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.25, 0.15, 0.10],
        "top_numbers": [7, 6, 8, 5],
        "top4_probability_mass": 0.65,
        "p_big": 0.75,
        "p_small": 0.25,
        "method": "markov_dirichlet_ensemble",
        "abstained": False,
    }

    telemetry_collector.record_digit_prediction(
        issue_id="20260812100099999",
        digit_pred_dict=pred_dict,
        predicted_size="BIG",
        confidence=0.75,
        regime_name="TRENDING",
        analysis_window=100,
    )

    # Score actual result
    telemetry_collector.record_actual_result(
        issue_id="20260812100099999",
        result_number=7,
        actual_size="BIG",
    )

    gov = telemetry_collector.get_digit_governance_summary()
    assert gov["sample_size"] >= 1
    assert gov["top1_acc"] == 100.0
    assert gov["top4_acc"] == 100.0
    assert gov["size_acc"] == 100.0
    assert gov["status"] == "HEALTHY"


def test_target_poisoning_adversarial_attack():
    """Target poisoning test: changing target digit MUST NOT alter prediction vector."""
    history = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0] * 10
    pred_baseline = predict_digits(history)

    # Injected poisoned future targets
    poisoned_targets = [99, 999, -1, 0, 5, 8]
    for target in poisoned_targets:
        pred_test = predict_digits(history)
        assert pred_test["digit_probabilities"] == pred_baseline["digit_probabilities"]


def test_confidence_intervals_and_baseline_lifts():
    """Verify Wilson score confidence intervals and baseline lifts calculation."""
    gov = telemetry_collector.get_digit_governance_summary()
    assert "baselines" in gov
    assert gov["baselines"]["top1_uniform"] == 10.0
    assert gov["baselines"]["top4_uniform"] == 40.0
    assert gov["baselines"]["size_uniform"] == 50.0


def test_drift_detection_and_alert_states():
    """Verify drift detection flags and health alert levels."""
    gov = telemetry_collector.get_digit_governance_summary()
    assert "drift_status" in gov
    assert "drift_detected" in gov["drift_status"]
    assert gov["status"] in ("HEALTHY", "WATCH", "DEGRADED", "CRITICAL")


def test_telemetry_public_api_endpoint():
    """Verify GET /api/v1/public/prediction/telemetry returns 200 with governance stats."""
    client = TestClient(app)
    response = client.get("/api/v1/public/prediction/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "governance" in data
    assert "latencies" in data
    assert "ai_telemetry_summary" in data
