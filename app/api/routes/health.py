"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.services.health_service import get_health, get_detailed_health

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_session)):
    """Basic health check endpoint."""
    return await get_health(session)


@router.get("/health/detailed")
async def detailed_health_check(session: AsyncSession = Depends(get_session)):
    """Detailed health check with diagnostics."""
    return await get_detailed_health(session)
