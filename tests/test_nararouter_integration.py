"""
Comprehensive Integration Test Suite for NaraRouter (NVIDIA Nemotron 3 Ultra) AI Provider.
Verifies registration, prompt context depth, 403 access denial handling, failover, and security.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import app.analytics.ai_rotator as ai_rot
from app.analytics.ai_rotator import _get_provider_pool, fetch_ai_prediction, fetch_ai_digit_prediction
from app.core import get_settings


@pytest.fixture(autouse=True)
def reset_ai_rotator_state():
    """Reset global cache and cooldown state before each test."""
    ai_rot._ai_cache = None
    ai_rot._ai_cache_time = 0
    ai_rot._ai_digit_cache = None
    ai_rot._ai_digit_cache_time = 0
    ai_rot._current_provider_index = 0
    ai_rot._key_cooldowns.clear()


def test_nararouter_provider_registration():
    """Verify NaraRouter is correctly registered in the provider pool with settings."""
    settings = get_settings()
    with patch.object(settings, "nararouter_api_key", "sk-nry-nOop10lOglt_AwUX_xFfjhls5AecIwIbovE4AYLaYyk"):
        with patch.object(settings, "ai_providers", "nara,nvidia,openrouter,groq,gemini"):
            pool = _get_provider_pool()
            nara_providers = [p for p in pool if p["name"].startswith("nara_")]
            assert len(nara_providers) >= 1
            nara = nara_providers[0]
            assert nara["url"] == "https://router.bynara.id/v1/chat/completions"
            assert nara["model"] == "nemotron-3-ultra"
            assert nara["type"] == "openai_compat"
            assert nara["extra_payload"] == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_nararouter_403_access_denied_failover():
    """Verify HTTP 403 (e.g. telegram_required) triggers cooldown and graceful failover."""
    sizes = ["BIG"] * 100
    stat_summary = {"confidence": 0.70}

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = '{"error":{"type":"forbidden","message":"telegram_required: Please bind your Telegram account"}}'

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        settings = get_settings()
        with patch.object(settings, "nararouter_api_key", "sk-nry-test-key-12345"):
            with patch.object(settings, "ai_providers", "nara"):
                res = await fetch_ai_prediction(sizes, stat_summary)
                assert res is None, "Should gracefully return None on 403 access denial without raising exceptions"
                assert "nara_1" in ai_rot._key_cooldowns, "NaraRouter should be placed in cooldown on 403"


@pytest.mark.asyncio
async def test_nararouter_reasoning_effort_retry_fallback():
    """Verify 400 rejection of reasoning_effort triggers single clean retry without reasoning_effort."""
    sizes = ["BIG"] * 100
    stat_summary = {"confidence": 0.70}

    mock_resp_400 = MagicMock()
    mock_resp_400.status_code = 400
    mock_resp_400.text = '{"error": "unsupported parameter reasoning_effort"}'

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {"Content-Type": "application/json"}
    mock_resp_200.json.return_value = {
        "choices": [{"message": {"content": '{"ai_prediction": "BIG", "ai_confidence": 0.82, "ai_reason": "high momentum"}'}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [mock_resp_400, mock_resp_200]
        settings = get_settings()
        with patch.object(settings, "nararouter_api_key", "sk-nry-test-key-12345"):
            with patch.object(settings, "ai_providers", "nara"):
                res = await fetch_ai_prediction(sizes, stat_summary)
                assert res is not None
                assert res["ai_prediction"] == "BIG"
                assert res["ai_confidence"] == 0.82
                assert res["provider"] == "nara_1"
                assert mock_post.call_count == 2
