"""Analytics service — orchestrates analytics calculations."""

from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.frequency import calculate_all_frequencies, calculate_frequency
from app.analytics.streaks import calculate_streaks
from app.analytics.transitions import calculate_transitions
from app.analytics.rolling import calculate_rolling_stats
from app.analytics.distribution import calculate_distribution
from app.analytics.anomaly import detect_anomalies
from app.analytics.prediction_engine import generate_prediction
from app.services.result_service import get_latest_result, get_total_count
from app.services.cache_service import cache
from app.core import get_settings


async def get_summary(session: AsyncSession) -> dict:
    """Get complete analytics summary."""
    cached = cache.get("summary")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    latest = await get_latest_result(session)
    total = await get_total_count(session)
    freq_100 = await calculate_frequency(session, 100)
    streaks = await calculate_streaks(session, 200)
    prediction = await generate_prediction(session, 500)
    anomalies = await detect_anomalies(session, 200)

    result = {
        "total_records": total,
        "latest_result": latest,
        "frequency_last_100": freq_100,
        "current_streak": {
            "size": streaks.get("current_size"),
            "length": streaks.get("current_streak"),
        },
        "prediction": {
            "next": prediction.get("prediction"),
            "confidence": prediction.get("confidence"),
            "confidence_level": prediction.get("confidence_level"),
            "agreeing_indicators": prediction.get("agreeing_indicators"),
            "label": "STATISTICAL ANALYSIS — NOT A GUARANTEE",
        },
        "anomaly_status": anomalies.get("status"),
        "data_updated_at": latest.get("observed_at") if latest else None,
        "api_generated_at": now.isoformat(),
    }

    cache.set("summary", result, settings.cache_summary_ttl)
    return result


async def get_frequency_stats(session: AsyncSession) -> dict:
    """Get frequency analysis for all windows."""
    cached = cache.get("frequency")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    frequencies = await calculate_all_frequencies(session)

    result = {
        "frequencies": frequencies,
        "api_generated_at": now.isoformat(),
        "label": "HISTORICAL STATISTICS",
    }

    cache.set("frequency", result, settings.cache_analytics_ttl)
    return result


async def get_streak_stats(session: AsyncSession) -> dict:
    """Get streak analysis."""
    cached = cache.get("streaks")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    streaks = await calculate_streaks(session)

    result = {
        "streaks": streaks,
        "api_generated_at": now.isoformat(),
        "label": "HISTORICAL STATISTICS",
    }

    cache.set("streaks", result, settings.cache_analytics_ttl)
    return result


async def get_transition_stats(session: AsyncSession) -> dict:
    """Get transition analysis."""
    cached = cache.get("transitions")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    transitions = await calculate_transitions(session)

    result = {
        "transitions": transitions,
        "api_generated_at": now.isoformat(),
        "label": "HISTORICAL STATISTICS",
    }

    cache.set("transitions", result, settings.cache_analytics_ttl)
    return result


async def get_rolling_stats(session: AsyncSession) -> dict:
    """Get rolling window statistics."""
    cached = cache.get("rolling")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    rolling = await calculate_rolling_stats(session)

    result = {
        "rolling_stats": rolling,
        "api_generated_at": now.isoformat(),
        "label": "HISTORICAL STATISTICS",
    }

    cache.set("rolling", result, settings.cache_analytics_ttl)
    return result


async def get_anomaly_stats(session: AsyncSession) -> dict:
    """Get anomaly detection results."""
    cached = cache.get("anomalies")
    if cached:
        return cached

    settings = get_settings()
    now = datetime.now(timezone.utc)

    anomalies = await detect_anomalies(session)

    result = {
        "anomalies": anomalies,
        "api_generated_at": now.isoformat(),
        "label": "HISTORICAL STATISTICS",
    }

    cache.set("anomalies", result, settings.cache_analytics_ttl)
    return result


async def get_prediction(session: AsyncSession) -> dict:
    """Get prediction from the event-driven pipeline."""
    from app.services.prediction_pipeline import pipeline

    prediction = pipeline.get_current_prediction()

    # If pipeline has no prediction yet, attempt a force refresh
    if prediction.get("status") == "INSUFFICIENT_DATA" and not prediction.get("upcoming_issue_id"):
        await pipeline.force_refresh()
        prediction = pipeline.get_current_prediction()

    return prediction

