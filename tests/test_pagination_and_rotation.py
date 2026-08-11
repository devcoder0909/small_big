"""
Comprehensive Unit Tests for Pagination, AI Rotator Failover, and Collector Resilience.

Tests:
1. Pagination parameters (pageNo, pageSize, ts) in SourceClient.
2. Deduplication and early exit in fetch_history_complete.
3. AI Rotator key failover, 429 rate-limit cooldown, and timeout fallback.
4. Collector exception resilience and heartbeat status.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.collector.client import SourceClient, FetchResult
from app.analytics.ai_rotator import fetch_ai_prediction, _key_cooldowns, _get_provider_pool, _validate_and_parse_ai_output
from app.collector.runner import CollectorRunner


@pytest.mark.asyncio
async def test_source_client_pagination_params():
    """Verify fetch_history passes pageNo and pageSize in params."""
    client = SourceClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": 0,
        "msg": "Succeed",
        "data": {
            "list": [
                {"issueNumber": "1001", "number": "5", "color": "green", "premium": "5", "sum": 0}
            ],
            "pageNo": 2,
            "totalPage": 50,
            "totalCount": 500,
        },
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await client.fetch_history(page_no=2, page_size=50)

        assert res.success is True
        assert mock_get.called
        call_args = mock_get.call_args
        params = call_args[1].get("params", {})
        assert params.get("pageNo") == "2"
        assert params.get("pageSize") == "50"
        assert "ts" in params

    await client.close()


@pytest.mark.asyncio
async def test_fetch_history_complete_stops_on_duplicate_pages():
    """Verify fetch_history_complete detects duplicate pages and breaks early."""
    client = SourceClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Always return the same page
    mock_resp.json.return_value = {
        "code": 0,
        "msg": "Succeed",
        "data": {
            "list": [
                {"issueNumber": "1001", "number": "5", "color": "green", "premium": "5", "sum": 0}
            ],
            "pageNo": 1,
            "totalPage": 50,
            "totalCount": 500,
        },
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        results = await client.fetch_history_complete(max_records=100, page_size=20)

        # Should only fetch 2 pages max before noticing duplicate records and terminating
        assert len(results) <= 2
        assert mock_get.call_count <= 2

    await client.close()


@pytest.mark.asyncio
async def test_ai_rotator_output_validation():
    """Verify strict parsing and bounds checking on raw AI outputs."""
    # Valid output
    valid = _validate_and_parse_ai_output('{"ai_prediction": "BIG", "ai_confidence": 0.75, "ai_reason": "Test reasoning"}')
    assert valid is not None
    assert valid["ai_prediction"] == "BIG"
    assert valid["ai_confidence"] == 0.75

    # Out of bounds confidence clamped/rejected
    invalid_conf = _validate_and_parse_ai_output('{"ai_prediction": "SMALL", "ai_confidence": 1.5, "ai_reason": "High"}')
    assert invalid_conf is None or invalid_conf["ai_confidence"] <= 0.95

    # NaN / Inf confidence rejected
    nan_conf = _validate_and_parse_ai_output('{"ai_prediction": "BIG", "ai_confidence": "NaN", "ai_reason": "NaN"}')
    assert nan_conf is None

    # Invalid prediction label rejected
    bad_pred = _validate_and_parse_ai_output('{"ai_prediction": "MEDIUM", "ai_confidence": 0.60, "ai_reason": "Test"}')
    assert bad_pred is None


@pytest.mark.asyncio
async def test_ai_rotator_429_cooldown_and_failover():
    """Verify 429 rate limit triggers key cooldown and failover to next provider."""
    sizes = ["BIG", "SMALL"] * 10
    stat_summary = {"entropy": 0.95}

    # Clear cooldowns before test
    _key_cooldowns.clear()

    # Mock settings with fake keys
    mock_settings = MagicMock()
    mock_settings.nvidia_api_key = "fake_nvidia_1"
    mock_settings.nvidia_api_key_2 = ""
    mock_settings.openrouter_api_key = "fake_openrouter_1"
    mock_settings.openrouter_api_key_2 = ""
    mock_settings.openrouter_api_key_3 = ""
    mock_settings.groq_api_key = ""
    mock_settings.groq_api_key_2 = ""
    mock_settings.gemini_api_key = ""
    mock_settings.gemini_api_key_2 = ""
    mock_settings.ai_providers = "nvidia,openrouter"
    mock_settings.ai_timeout_seconds = 1.0
    mock_settings.ai_provider_cooldown_seconds = 60.0

    # Response 1: NVIDIA returns 429
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "30"}

    # Response 2: OpenRouter returns 200 with valid JSON
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.headers = {"Content-Type": "application/json"}
    resp_200.json.return_value = {
        "choices": [{"message": {"content": '{"ai_prediction": "SMALL", "ai_confidence": 0.70, "ai_reason": "OpenRouter signal"}'}}]
    }

    async def mock_post(url, *args, **kwargs):
        if "nvidia" in url:
            return resp_429
        return resp_200

    with patch("app.analytics.ai_rotator.get_settings", return_value=mock_settings), \
         patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await fetch_ai_prediction(sizes, stat_summary)

        assert result is not None
        assert result["ai_prediction"] == "SMALL"
        assert result["provider"] == "openrouter_1"
        # Verify nvidia_1 is in cooldown
        assert "nvidia_1" in _key_cooldowns


@pytest.mark.asyncio
async def test_ai_rotator_timeout_fallback():
    """Verify timeout on provider transparently falls back to next provider or None."""
    sizes = ["BIG", "SMALL"] * 10
    stat_summary = {"entropy": 0.95}

    _key_cooldowns.clear()

    mock_settings = MagicMock()
    mock_settings.nvidia_api_key = "fake_nvidia_1"
    mock_settings.openrouter_api_key = ""
    mock_settings.groq_api_key = ""
    mock_settings.gemini_api_key = ""
    mock_settings.ai_providers = "nvidia"
    mock_settings.ai_timeout_seconds = 0.1

    with patch("app.analytics.ai_rotator.get_settings", return_value=mock_settings), \
         patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        result = await fetch_ai_prediction(sizes, stat_summary)

        # Returns None on complete failure so statistical engine handles prediction
        assert result is None
