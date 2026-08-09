"""Admin endpoints — protected administrative operations."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session, verify_admin_key
from app.services.recovery_service import recover_missing_records, detect_gaps
from app.services.cache_service import cache

router = APIRouter(tags=["admin"], dependencies=[Depends(verify_admin_key)])


@router.get("/collector-status")
async def collector_status(session: AsyncSession = Depends(get_session)):
    """Get collector status details."""
    from app.services.health_service import get_detailed_health
    return await get_detailed_health(session)


@router.post("/recover")
async def trigger_recovery(session: AsyncSession = Depends(get_session)):
    """Trigger manual recovery of missing records."""
    async with session.begin():
        return await recover_missing_records(session)


@router.get("/data-quality")
async def data_quality_report(session: AsyncSession = Depends(get_session)):
    """Get data quality report including gaps."""
    gaps = await detect_gaps(session, window=500)
    return {
        "gaps_detected": len(gaps),
        "gaps": gaps,
    }


@router.post("/recalculate")
async def recalculate_analytics():
    """Clear analytics cache to force recalculation."""
    cache.clear()
    return {"status": "cache_cleared", "message": "Analytics will be recalculated on next request"}
