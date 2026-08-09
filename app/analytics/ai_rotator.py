"""
AI API Key Rotator & Pattern Reasoning Service.

Rotates through multi-key free-tier providers (Groq, OpenRouter, Gemini)
to perform non-blocking LLM pattern analysis and boost prediction confidence.
"""

import os
import json
import time
import httpx

from app.core import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global state for rotation & caching
_current_provider_index = 0
_ai_cache: dict | None = None
_ai_cache_time: float = 0
CACHE_TTL = 8.0  # Cache AI predictions for 8 seconds to respect free-tier quotas


def _get_provider_pool() -> list[dict]:
    """Get active multi-key provider pool loaded from environment settings."""
    settings = get_settings()

    groq_keys = [
        settings.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
        settings.groq_api_key_2 or os.environ.get("GROQ_API_KEY_2", ""),
    ]
    openrouter_keys = [
        settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        settings.openrouter_api_key_2 or os.environ.get("OPENROUTER_API_KEY_2", ""),
        settings.openrouter_api_key_3 or os.environ.get("OPENROUTER_API_KEY_3", ""),
    ]
    gemini_keys = [
        settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", ""),
    ]

    pool = []

    # Groq keys
    for i, k in enumerate(groq_keys):
        if k:
            pool.append({
                "name": f"groq_{i+1}",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "key": k,
                "model": "llama-3.1-8b-instant",
                "type": "openai_compat",
            })

    # OpenRouter keys
    for i, k in enumerate(openrouter_keys):
        if k:
            pool.append({
                "name": f"openrouter_{i+1}",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": k,
                "model": "openai/gpt-4o-mini",
                "type": "openai_compat",
            })

    # Gemini keys
    for i, k in enumerate(gemini_keys):
        if k:
            pool.append({
                "name": f"gemini_{i+1}",
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                "key": k,
                "model": "gemini-2.0-flash",
                "type": "gemini",
            })

    return pool


async def fetch_ai_prediction(sizes: list[str], stat_summary: dict) -> dict | None:
    """
    Fetch an AI pattern reasoning signal using multi-key rotation.

    Args:
        sizes: Recent draw sizes (e.g. ['SMALL', 'BIG', ...]).
        stat_summary: Summary of local 8-indicator statistical scores.

    Returns:
        Dict with AI prediction, confidence, and reasoning, or None if unavailable.
    """
    global _current_provider_index, _ai_cache, _ai_cache_time

    # Return cached prediction if fresh
    now_mono = time.monotonic()
    if _ai_cache and (now_mono - _ai_cache_time) < CACHE_TTL:
        return _ai_cache

    if not sizes or len(sizes) < 10:
        return None

    providers = _get_provider_pool()
    if not providers:
        return None

    recent_sequence = ", ".join(sizes[:40])
    prompt = (
        f"You are a world-class mathematical pattern analyst specializing in binary sequence forecasting.\n"
        f"Your job: analyze the sequence and statistical indicators below, then predict the NEXT outcome.\n\n"
        f"RECENT 40 DRAWS (newest first): [{recent_sequence}]\n\n"
        f"LOCAL 10-INDICATOR STATISTICAL ENGINE OUTPUT:\n{json.dumps(stat_summary, indent=2)}\n\n"
        f"ANALYSIS REQUIRED:\n"
        f"1. Check streak exhaustion patterns\n"
        f"2. Check frequency imbalance (law of large numbers reversion)\n"
        f"3. Check Markov transition probabilities\n"
        f"4. Check N-gram pattern repetitions\n"
        f"5. Check regime (structured vs chaotic) from entropy\n"
        f"6. Weigh indicator consensus vs disagreement\n\n"
        f"Respond ONLY with a JSON object in this EXACT format, with no markdown or text:\n"
        f'{{"ai_prediction": "SMALL" or "BIG", "ai_confidence": 0.55 to 0.95, "ai_reason": "concise 1-sentence reasoning"}}'
    )

    num_providers = len(providers)
    start_idx = _current_provider_index

    for step in range(num_providers):
        idx = (start_idx + step) % num_providers
        provider = providers[idx]

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                if provider["type"] == "openai_compat":
                    resp = await client.post(
                        provider["url"],
                        headers={
                            "Authorization": f"Bearer {provider['key']}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": provider["model"],
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2,
                            "max_tokens": 100,
                        },
                    )
                else:
                    # Gemini format
                    resp = await client.post(
                        f"{provider['url']}?key={provider['key']}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 100},
                        },
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""

                    if provider["type"] == "openai_compat":
                        raw_text = data["choices"][0]["message"]["content"].strip()
                    else:
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                    # Sanitize markdown codeblocks if present
                    if "```" in raw_text:
                        raw_text = raw_text.split("```")[1].replace("json", "").strip()

                    parsed = json.loads(raw_text)

                    if "ai_prediction" in parsed and parsed["ai_prediction"] in ("SMALL", "BIG"):
                        result = {
                            "ai_prediction": parsed["ai_prediction"],
                            "ai_confidence": min(0.95, max(0.50, float(parsed.get("ai_confidence", 0.60)))),
                            "ai_reason": str(parsed.get("ai_reason", "AI pattern match")),
                            "provider": provider["name"],
                            "model": provider["model"],
                        }

                        # Update cache and advance provider index for next cycle
                        _ai_cache = result
                        _ai_cache_time = now_mono
                        _current_provider_index = (idx + 1) % num_providers

                        logger.info("ai_prediction_success", provider=provider["name"], result=result)
                        return result

                logger.warning(
                    "ai_provider_non_200",
                    provider=provider["name"],
                    status=resp.status_code,
                )

        except Exception as e:
            logger.warning("ai_provider_error", provider=provider["name"], error=str(e))

    return None
