"""Results endpoints — verified observed game results."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session, verify_api_key
from app.services.result_service import get_latest_result, get_results, get_result_by_issue
from app.services.cache_service import cache
from app.core import get_settings

router = APIRouter(tags=["results"], dependencies=[Depends(verify_api_key)])


@router.get("/latest")
async def latest_result(session: AsyncSession = Depends(get_session)):
    """
    Get the latest observed game result.

    Returns the most recent ACTUAL RESULT from the source API.
    This is a verified observation, not a prediction.
    """
    settings = get_settings()
    cached = cache.get("latest")
    if cached:
        return cached

    result = await get_latest_result(session)
    if not result:
        raise HTTPException(status_code=404, detail="No results available yet")

    cache.set("latest", result, settings.cache_latest_ttl)
    return result


@router.get("/results")
async def list_results(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """Get paginated game results."""
    return await get_results(session, limit=limit, offset=offset)


@router.get("/results/{issue_id}")
async def get_result(
    issue_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific result by issue ID."""
    result = await get_result_by_issue(session, issue_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
    return result
