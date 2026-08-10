"""
Test Suite for Production Truth Service & Analytics Endpoint.

Verifies:
1. Storage alarm level calculations (50%, 65%, 75%, 85%, 90% thresholds).
2. Wilson Score Interval math & Binomial p-value accuracy.
3. Calibration bucket filtering with minimum N >= 30 sample threshold.
4. Production truth JSON endpoint schema.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.production_truth_service import (
    calculate_wilson_score_interval,
    calculate_binomial_p_value,
    get_storage_alarm_level,
    generate_production_truth_report,
)


def test_storage_alarm_levels():
    info = get_storage_alarm_level(1_000_000_000, 6_000_000_000)
    assert info["alarm_level"] == "INFO"

    info50 = get_storage_alarm_level(3_100_000_000, 6_000_000_000)
    assert info50["alarm_level"] == "INFO_WARNING"

    warn65 = get_storage_alarm_level(4_000_000_000, 6_000_000_000)
    assert warn65["alarm_level"] == "WARNING"

    high75 = get_storage_alarm_level(4_600_000_000, 6_000_000_000)
    assert high75["alarm_level"] == "HIGH_WARNING"

    crit85 = get_storage_alarm_level(5_200_000_000, 6_000_000_000)
    assert crit85["alarm_level"] == "CRITICAL"

    emerg90 = get_storage_alarm_level(5_500_000_000, 6_000_000_000)
    assert emerg90["alarm_level"] == "EMERGENCY"


def test_wilson_score_and_p_value():
    # 60 wins out of 100 -> 60%
    lower, upper = calculate_wilson_score_interval(60, 100)
    assert 49.0 <= lower <= 52.0
    assert 68.0 <= upper <= 70.0

    p_val = calculate_binomial_p_value(60, 100, p0=0.50)
    assert p_val < 0.05  # Statistically significant edge over 50% baseline


@pytest.mark.asyncio
async def test_generate_production_truth_report_structure(db_session):
    report = await generate_production_truth_report(db_session)

    assert "status" in report
    assert report["status"] in ("PARTIALLY_VERIFIED", "VERIFIED_LIVE_PRODUCTION", "NOT_VERIFIED")
    assert "database" in report
    assert "storage" in report
    assert "pipeline" in report
    assert "accuracy" in report
    assert "calibration_buckets" in report
    assert "baselines" in report

    # Check calibration bucket threshold (insufficient sample handling)
    for bucket_name, bucket_info in report["calibration_buckets"].items():
        assert "status" in bucket_info
        if bucket_info["sample_count"] < 30:
            assert bucket_info["status"] == "INSUFFICIENT_SAMPLE"
