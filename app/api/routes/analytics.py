"""Analytics endpoints — historical statistics and prediction."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session, verify_api_key
from app.services.analytics_service import (
    get_summary,
    get_frequency_stats,
    get_streak_stats,
    get_transition_stats,
    get_rolling_stats,
    get_anomaly_stats,
    get_prediction,
)

router = APIRouter(tags=["analytics"], dependencies=[Depends(verify_api_key)])


@router.get("/stats/summary")
async def stats_summary(session: AsyncSession = Depends(get_session)):
    """
    Complete analytics summary.

    Includes latest result, frequency analysis, current streak,
    prediction, and anomaly status.
    """
    return await get_summary(session)


@router.get("/stats/frequency")
async def stats_frequency(session: AsyncSession = Depends(get_session)):
    """
    Historical frequency analysis.

    Small/Big distribution across windows: 20, 50, 100, 500, 1000, all-time.
    HISTORICAL STATISTICS — not prediction.
    """
    return await get_frequency_stats(session)


@router.get("/stats/streaks")
async def stats_streaks(session: AsyncSession = Depends(get_session)):
    """
    Historical streak analysis.

    Current streak, longest streaks, average streak lengths.
    HISTORICAL STATISTICS — a long streak does NOT guarantee reversal.
    """
    return await get_streak_stats(session)


@router.get("/stats/transitions")
async def stats_transitions(session: AsyncSession = Depends(get_session)):
    """
    Historical transition analysis.

    Small→Small, Small→Big, Big→Small, Big→Big frequencies and percentages.
    HISTORICAL STATISTICS — past transitions do NOT determine future outcomes.
    """
    return await get_transition_stats(session)


@router.get("/stats/rolling")
async def stats_rolling(session: AsyncSession = Depends(get_session)):
    """
    Rolling window statistics.

    Sliding window analysis over recent results.
    HISTORICAL STATISTICS.
    """
    return await get_rolling_stats(session)


@router.get("/stats/anomalies")
async def stats_anomalies(session: AsyncSession = Depends(get_session)):
    """
    Anomaly detection results.

    Statistically unusual patterns: NORMAL, WATCH, or ANOMALY.
    HISTORICAL STATISTICS — not gambling advice.
    """
    return await get_anomaly_stats(session)


@router.get("/stats/prediction")
async def stats_prediction(session: AsyncSession = Depends(get_session)):
    """
    Statistical prediction for next Small/Big outcome.

    IMPORTANT: This is STATISTICAL ANALYSIS based on historical patterns.
    Each game round is an independent event.
    Past patterns do NOT guarantee future outcomes.
    """
    return await get_prediction(session)


@router.get("/v3-metrics")
async def get_v3_metrics():
    """
    V3 Adaptive Edge Discovery Internal Dashboard Metrics.

    Exposes champion strategy performance, regime statistics,
    abstention rates, and Brier score tracking.
    """
    from app.analytics.champion_selector import champion_selector
    return {
        "status": "HEALTHY",
        "mode": "V3_ADAPTIVE_EDGE_DISCOVERY",
        "champion_selector": champion_selector.get_metrics_summary(),
        "disclaimer": "Internal OOS performance metrics for champion/challenger strategy evaluation.",
    }
