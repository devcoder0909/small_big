"""
AI API Provider Rotator & Failover Ensemble Service.

Integrates NVIDIA NIM (Nemotron), OpenRouter, Groq, and Gemini LLM providers
with failover ordering, rate-limit cooldowns, strict output validation, and telemetry metrics.
"""

import os
import re
import json
import time
import math
import asyncio
import httpx

from app.core import get_settings
from app.core.logging import get_logger
from app.analytics.telemetry import telemetry_collector

logger = get_logger(__name__)

# Global state for rotation, caching & key cooldowns
_current_provider_index = 0
_ai_cache: dict | None = None
_ai_cache_time: float = 0
_key_cooldowns: dict[str, float] = {}

CACHE_TTL = 8.0  # Cache AI predictions for 8 seconds to respect rate limits


def _is_valid_api_key(key: str) -> bool:
    if not key:
        return False
    k = key.strip().lower()
    if any(k.startswith(prefix) for prefix in ("your_", "mock_", "dummy_", "placeholder_")):
        return False
    if len(k) < 8:
        return False
    return True


def _get_provider_pool() -> list[dict]:
    """
    Get active provider pool dynamically configured from settings.
    Providers with missing or placeholder API keys are automatically disabled.
    """
    settings = get_settings()

    nvidia_keys = [
        getattr(settings, "nvidia_api_key", ""),
        getattr(settings, "nvidia_api_key_2", ""),
    ]
    nvidia_url = getattr(settings, "nvidia_base_url", "https://integrate.api.nvidia.com/v1")
    nvidia_model = getattr(settings, "nvidia_model", "nvidia/nemotron-3-ultra-550b-a55b")

    openrouter_url = getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
    openrouter_model = getattr(settings, "openrouter_model", "meta-llama/llama-3.1-70b-instruct")
    openrouter_keys = [
        getattr(settings, "openrouter_api_key", ""),
        getattr(settings, "openrouter_api_key_2", ""),
        getattr(settings, "openrouter_api_key_3", ""),
    ]

    groq_keys = [
        getattr(settings, "groq_api_key", ""),
        getattr(settings, "groq_api_key_2", ""),
    ]

    gemini_keys = [
        getattr(settings, "gemini_api_key", ""),
        getattr(settings, "gemini_api_key_2", ""),
    ]

    # Parse configured provider priority list (e.g. "nvidia,openrouter,groq,gemini")
    raw_order = getattr(settings, "ai_providers", "nvidia,openrouter,groq,gemini")
    priority_list = [p.strip().lower() for p in raw_order.split(",") if p.strip()]

    pool = []
    registered_names = set()

    for provider_name in priority_list:
        if "nvidia" in provider_name and "nvidia" not in registered_names:
            for i, k in enumerate(nvidia_keys):
                if _is_valid_api_key(k):
                    pool.append({
                        "name": f"nvidia_{i+1}",
                        "url": f"{nvidia_url.rstrip('/')}/chat/completions",
                        "key": k,
                        "model": nvidia_model,
                        "type": "openai_compat",
                    })
            registered_names.add("nvidia")

        elif "openrouter" in provider_name and "openrouter" not in registered_names:
            for i, k in enumerate(openrouter_keys):
                if _is_valid_api_key(k):
                    pool.append({
                        "name": f"openrouter_{i+1}",
                        "url": f"{openrouter_url.rstrip('/')}/chat/completions",
                        "key": k,
                        "model": openrouter_model,
                        "type": "openai_compat",
                    })
            registered_names.add("openrouter")

        elif "groq" in provider_name and "groq" not in registered_names:
            for i, k in enumerate(groq_keys):
                if _is_valid_api_key(k):
                    pool.append({
                        "name": f"groq_{i+1}",
                        "url": "https://api.groq.com/openai/v1/chat/completions",
                        "key": k,
                        "model": "llama-3.1-8b-instant",
                        "type": "openai_compat",
                    })
            registered_names.add("groq")

        elif "gemini" in provider_name and "gemini" not in registered_names:
            for i, k in enumerate(gemini_keys):
                if _is_valid_api_key(k):
                    pool.append({
                        "name": f"gemini_{i+1}",
                        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                        "key": k,
                        "model": "gemini-1.5-flash",
                        "type": "gemini",
                    })
            registered_names.add("gemini")

    return pool


def _validate_and_parse_ai_output(raw_text: str) -> dict | None:
    """
    Strict parsing and validation of untrusted AI response output.
    Rejects invalid prediction labels, non-numeric/out-of-bound confidence, NaN, infinity,
    malformed JSON, and prompt-injection patterns.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for p in parts[1:]:
            p_clean = p.replace("json", "").strip()
            if p_clean.startswith("{") and p_clean.endswith("}"):
                cleaned = p_clean
                break

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass

    if not isinstance(data, dict):
        return None

    pred = str(data.get("ai_prediction") or data.get("prediction") or "").upper().strip()
    if pred not in ("BIG", "SMALL"):
        return None

    raw_conf = data.get("ai_confidence") if "ai_confidence" in data else data.get("confidence")
    try:
        conf = float(raw_conf)
        if math.isnan(conf) or math.isinf(conf):
            return None
        if conf < 0.0 or conf > 1.0:
            return None
        conf = min(0.95, max(0.50, conf))
    except (ValueError, TypeError):
        return None

    reason = str(data.get("ai_reason") or data.get("reason") or "AI pattern analysis")

    # Anti-prompt-injection validation
    injection_keywords = ["ignore previous", "system prompt", "overwrite", "drop table", "<script>", "eval("]
    if any(kw in reason.lower() for kw in injection_keywords):
        reason = "AI pattern analysis"

    return {
        "ai_prediction": pred,
        "ai_confidence": round(conf, 4),
        "ai_reason": reason[:200],
    }


def _parse_sse_stream_chunks(stream_text: str) -> tuple[str, str]:
    """Parse Server-Sent Events (SSE) stream text for content and reasoning_content."""
    content_parts = []
    reasoning_parts = []
    for line in stream_text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            chunk_str = line[6:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk = json.loads(chunk_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    reasoning_parts.append(delta["reasoning_content"])
            except Exception:
                pass
    return "".join(content_parts), "".join(reasoning_parts)


async def fetch_ai_prediction(sizes: list[str], stat_summary: dict) -> dict | None:
    """
    Fetch an AI pattern reasoning signal using configured provider failover order.

    Args:
        sizes: Historical draw sizes prior to target period.
        stat_summary: Statistical indicator values prior to target period.

    Returns:
        Dict with validated prediction, confidence, reasoning, provider, and model, or None if failed.
    """
    global _current_provider_index, _ai_cache, _ai_cache_time

    settings = get_settings()
    now_mono = time.monotonic()

    # Return cached prediction if fresh
    if _ai_cache and (now_mono - _ai_cache_time) < CACHE_TTL:
        return _ai_cache

    if not sizes or len(sizes) < 10:
        return None

    providers = _get_provider_pool()
    if not providers:
        return None

    chronological_sequence = ", ".join(list(reversed(sizes[:40])))
    prompt = (
        "You are a world-class mathematical pattern analyst specializing in binary sequence forecasting.\n"
        "Your job: analyze the sequence and statistical indicators below, then predict the NEXT outcome.\n\n"
        f"RECENT 40 DRAWS (chronological order: oldest -> newest): [{chronological_sequence}]\n\n"
        f"LOCAL STATISTICAL ENGINE OUTPUT:\n{json.dumps(stat_summary, indent=2)}\n\n"
        "Respond ONLY with a JSON object in this EXACT format, with no markdown or text:\n"
        '{"ai_prediction": "SMALL" or "BIG", "ai_confidence": 0.55 to 0.95, "ai_reason": "concise 1-sentence reasoning"}'
    )

    num_providers = len(providers)
    start_idx = _current_provider_index
    timeout_sec = float(getattr(settings, "ai_timeout_seconds", 3.0))

    t_overall_start = time.monotonic()
    for step in range(num_providers):
        if (time.monotonic() - t_overall_start) >= timeout_sec:
            break
        idx = (start_idx + step) % num_providers
        provider = providers[idx]
        p_name = provider["name"]

        # Check rate-limit cooldown
        cooldown_until = _key_cooldowns.get(p_name, 0)
        if now_mono < cooldown_until:
            continue

        t_start = time.monotonic()
        req_duration_ms = 0.0
        success = False
        is_timeout = False
        status_category = "error"
        reasoning_content = None

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                if provider["type"] == "openai_compat":
                    payload = {
                        "model": provider["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "max_tokens": 200,
                    }
                    if "nemotron" in provider["model"].lower():
                        payload["reasoning_budget"] = 100

                    resp = await client.post(
                        provider["url"],
                        headers={
                            "Authorization": f"Bearer {provider['key']}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                else:
                    # Gemini API
                    resp = await client.post(
                        f"{provider['url']}?key={provider['key']}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
                        },
                    )

                req_duration_ms = round((time.monotonic() - t_start) * 1000.0, 2)
                status_category = f"{resp.status_code // 100}xx" if resp.status_code else "error"

                if resp.status_code == 200:
                    raw_text = ""
                    content_type = resp.headers.get("Content-Type", "")

                    if "text/event-stream" in content_type:
                        raw_text, reasoning_content = _parse_sse_stream_chunks(resp.text)
                    else:
                        data = resp.json()
                        if provider["type"] == "openai_compat":
                            choice = data.get("choices", [{}])[0]
                            msg = choice.get("message", {})
                            raw_text = msg.get("content", "") or ""
                            reasoning_content = msg.get("reasoning_content") or choice.get("reasoning_content")
                        else:
                            raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

                    validated = _validate_and_parse_ai_output(raw_text)

                    if validated:
                        success = True
                        result = {
                            "ai_prediction": validated["ai_prediction"],
                            "ai_confidence": validated["ai_confidence"],
                            "ai_reason": validated["ai_reason"],
                            "provider": provider["name"],
                            "model": provider["model"],
                        }
                        if reasoning_content:
                            result["reasoning_content"] = reasoning_content

                        _ai_cache = result
                        _ai_cache_time = now_mono
                        _current_provider_index = (idx + 1) % num_providers
                        _key_cooldowns.pop(p_name, None)

                        telemetry_collector.record_ai_request(
                            provider=provider["name"],
                            model=provider["model"],
                            request_started_at=t_start,
                            request_duration_ms=req_duration_ms,
                            success=True,
                            timeout=False,
                            http_status_category="2xx",
                            fallback_used=False,
                            ai_contribution_status="accepted",
                        )

                        logger.info("ai_prediction_success", provider=provider["name"], model=provider["model"])
                        return result

                elif resp.status_code == 429:
                    status_category = "429"
                    retry_after = resp.headers.get("Retry-After")
                    cooldown = float(getattr(settings, "ai_provider_cooldown_seconds", 60.0))
                    if retry_after:
                        try:
                            cooldown = float(retry_after)
                        except ValueError:
                            pass
                    _key_cooldowns[p_name] = now_mono + cooldown
                    logger.warning("ai_provider_rate_limited", provider=p_name, cooldown_seconds=cooldown)

                elif resp.status_code >= 500:
                    status_category = "5xx"
                    logger.warning("ai_provider_server_error", provider=p_name, status=resp.status_code)

                else:
                    logger.warning("ai_provider_non_200", provider=p_name, status=resp.status_code)

        except (httpx.TimeoutException, asyncio.TimeoutError):
            is_timeout = True
            status_category = "timeout"
            req_duration_ms = round((time.monotonic() - t_start) * 1000.0, 2)
            logger.warning("ai_provider_timeout", provider=p_name, duration_ms=req_duration_ms)

        except Exception as e:
            req_duration_ms = round((time.monotonic() - t_start) * 1000.0, 2)
            logger.warning("ai_provider_error", provider=p_name, error=str(e))

        # Record telemetry for failed attempt
        telemetry_collector.record_ai_request(
            provider=p_name,
            model=provider["model"],
            request_started_at=t_start,
            request_duration_ms=req_duration_ms,
            success=False,
            timeout=is_timeout,
            http_status_category=status_category,
            fallback_used=True,
            ai_contribution_status="failed",
        )

    return None


_ai_digit_cache: dict | None = None
_ai_digit_cache_time: float = 0


def _validate_and_parse_ai_digit_output(raw_text: str) -> dict | None:
    """Strict validation for AI digit output (0-9 integer, confidence, top 3)."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for p in parts[1:]:
            p_clean = p.replace("json", "").strip()
            if p_clean.startswith("{") and p_clean.endswith("}"):
                cleaned = p_clean
                break

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass

    if not isinstance(data, dict):
        return None

    raw_digit = data.get("ai_digit_prediction")
    if raw_digit is None:
        raw_digit = data.get("digit_prediction") or data.get("predicted_digit")

    try:
        pred_digit = int(raw_digit)
        if pred_digit < 0 or pred_digit > 9:
            return None
    except (ValueError, TypeError):
        return None

    raw_conf = data.get("ai_digit_confidence") if "ai_digit_confidence" in data else data.get("confidence", 0.15)
    try:
        conf = float(raw_conf)
        if math.isnan(conf) or math.isinf(conf) or conf < 0.0 or conf > 1.0:
            return None
        conf = min(0.95, max(0.10, conf))
    except (ValueError, TypeError):
        conf = 0.15

    raw_top3 = data.get("ai_top_3") or data.get("top_3") or [pred_digit]
    top_3 = []
    if isinstance(raw_top3, list):
        for item in raw_top3:
            try:
                d_val = int(item)
                if 0 <= d_val <= 9 and d_val not in top_3:
                    top_3.append(d_val)
            except (ValueError, TypeError):
                pass
    if pred_digit not in top_3:
        top_3.insert(0, pred_digit)
    top_3 = top_3[:3]

    reason = str(data.get("ai_reason") or "AI digit pattern analysis")
    injection_keywords = ["ignore previous", "system prompt", "overwrite", "drop table", "<script>", "eval("]
    if any(kw in reason.lower() for kw in injection_keywords):
        reason = "AI digit pattern analysis"

    return {
        "ai_digit_prediction": pred_digit,
        "ai_digit_confidence": round(conf, 4),
        "ai_top_3": top_3,
        "ai_reason": reason[:200],
    }


async def fetch_ai_digit_prediction(numbers: list[int], sizes: list[str], stat_summary: dict) -> dict | None:
    """Fetch AI digit hypothesis signal across provider rotation pool."""
    global _current_provider_index, _ai_digit_cache, _ai_digit_cache_time

    settings = get_settings()
    now_mono = time.monotonic()

    if _ai_digit_cache and (now_mono - _ai_digit_cache_time) < CACHE_TTL:
        return _ai_digit_cache

    if not numbers or len(numbers) < 10:
        return None

    providers = _get_provider_pool()
    if not providers:
        return None

    digit_seq = ", ".join(str(n) for n in list(reversed(numbers[:40])))
    prompt = (
        "You are a quantitative mathematical analyst specializing in single-digit (0-9) pattern forecasting.\n"
        "Your task: analyze the digit sequence below, then predict the NEXT single digit (0-9).\n\n"
        f"RECENT 40 DIGIT DRAWS (oldest -> newest): [{digit_seq}]\n\n"
        f"LOCAL STATISTICAL SUMMARY:\n{json.dumps(stat_summary, indent=2)}\n\n"
        "Respond ONLY with a JSON object in this EXACT format, with no markdown or extra text:\n"
        '{"ai_digit_prediction": 7, "ai_digit_confidence": 0.15, "ai_top_3": [7, 8, 6], "ai_reason": "concise 1-sentence reasoning"}'
    )

    num_providers = len(providers)
    start_idx = _current_provider_index
    timeout_sec = float(getattr(settings, "ai_timeout_seconds", 3.0))

    t_overall_start = time.monotonic()
    for step in range(num_providers):
        if (time.monotonic() - t_overall_start) >= timeout_sec:
            break
        idx = (start_idx + step) % num_providers
        provider = providers[idx]
        p_name = provider["name"]

        cooldown_until = _key_cooldowns.get(p_name, 0)
        if now_mono < cooldown_until:
            continue

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                if provider["type"] == "openai_compat":
                    payload = {
                        "model": provider["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 150,
                    }
                    resp = await client.post(
                        provider["url"],
                        headers={
                            "Authorization": f"Bearer {provider['key']}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                else:
                    resp = await client.post(
                        f"{provider['url']}?key={provider['key']}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 150},
                        },
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    if provider["type"] == "openai_compat":
                        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                    else:
                        raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

                    validated = _validate_and_parse_ai_digit_output(raw_text)
                    if validated:
                        result = {
                            "ai_digit_prediction": validated["ai_digit_prediction"],
                            "ai_digit_confidence": validated["ai_digit_confidence"],
                            "ai_top_3": validated["ai_top_3"],
                            "ai_reason": validated["ai_reason"],
                            "provider": provider["name"],
                            "model": provider["model"],
                        }
                        _ai_digit_cache = result
                        _ai_digit_cache_time = now_mono
                        _current_provider_index = (idx + 1) % num_providers
                        return result
        except Exception as err:
            logger.warning("ai_digit_provider_error", provider=p_name, error=str(err))

    return None

