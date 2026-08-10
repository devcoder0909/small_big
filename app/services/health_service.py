"""Health monitoring service."""

from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_heartbeat import SystemHeartbeat
from app.models.game_result import GameResult
from app.models.data_quality import DataQualityEvent
from app.models.source_request import SourceRequest
from app.core import get_settings, get_build_commit
from app.core.database import check_db_connection


async def get_health(session: AsyncSession) -> dict:
    """Get basic health status."""
    db_connected = await check_db_connection()

    # Get heartbeat
    result = await session.execute(
        select(SystemHeartbeat).where(SystemHeartbeat.service_name == "collector")
    )
    heartbeat = result.scalar_one_or_none()

    # Get total records
    count_result = await session.execute(
        select(func.count()).select_from(GameResult)
    )
    total_records = count_result.scalar() or 0

    collector_status = "unknown"
    last_fetch = None
    last_new = None

    if heartbeat:
        collector_status = heartbeat.status or "unknown"
        last_fetch = heartbeat.last_successful_fetch
        last_new = heartbeat.last_new_record

        # Check staleness
        settings = get_settings()
        if last_fetch:
            # Ensure timezone-aware datetime comparison
            last_fetch_utc = last_fetch.replace(tzinfo=timezone.utc) if last_fetch.tzinfo is None else last_fetch
            age = (datetime.now(timezone.utc) - last_fetch_utc).total_seconds()
            if age > settings.health_degraded_threshold_seconds:
                collector_status = "DEGRADED"

    c_status_lower = collector_status.lower() if collector_status else "unknown"

    if not db_connected:
        overall = "unhealthy"
    elif c_status_lower in ("healthy", "running", "starting"):
        overall = "healthy"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "build_commit": get_build_commit(),
        "collector": c_status_lower,
        "database": "connected" if db_connected else "disconnected",
        "last_successful_fetch": last_fetch.isoformat() if last_fetch else None,
        "last_new_record": last_new.isoformat() if last_new else None,
        "records_total": total_records,
    }


async def get_detailed_health(session: AsyncSession) -> dict:
    """Get detailed health status with diagnostics."""
    basic = await get_health(session)

    # Get heartbeat details
    result = await session.execute(
        select(SystemHeartbeat).where(SystemHeartbeat.service_name == "collector")
    )
    heartbeat = result.scalar_one_or_none()

    # Recent errors
    error_result = await session.execute(
        select(func.count()).select_from(DataQualityEvent)
        .where(DataQualityEvent.severity.in_(["ERROR", "CRITICAL"]))
    )
    error_count = error_result.scalar() or 0

    # Missing issues
    missing_result = await session.execute(
        select(func.count()).select_from(DataQualityEvent)
        .where(DataQualityEvent.event_type == "missing_issue")
        .where(DataQualityEvent.resolved == False)
    )
    missing_count = missing_result.scalar() or 0

    # Duplicate count from heartbeat
    duplicate_count = heartbeat.total_duplicates if heartbeat else 0

    # Last request stats
    last_req = await session.execute(
        select(SourceRequest)
        .order_by(SourceRequest.id.desc())
        .limit(1)
    )
    last_request = last_req.scalar_one_or_none()

    source_latency = last_request.response_time_ms if last_request else None

    # DB latency test
    import time
    start = time.monotonic()
    await session.execute(select(func.count()).select_from(GameResult))
    db_latency_ms = int((time.monotonic() - start) * 1000)

    return {
        **basic,
        "detailed": {
            "source_latency_ms": source_latency,
            "database_latency_ms": db_latency_ms,
            "collector_uptime_seconds": heartbeat.uptime_seconds if heartbeat else 0,
            "total_errors": error_count,
            "total_duplicates": duplicate_count,
            "missing_issues": missing_count,
            "last_heartbeat": heartbeat.last_heartbeat.isoformat() if heartbeat and heartbeat.last_heartbeat else None,
        },
        "api_generated_at": datetime.now(timezone.utc).isoformat(),
    }
