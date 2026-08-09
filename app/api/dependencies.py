"""API dependencies — authentication and database session injection."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_settings
from app.core.database import async_session_factory

# API Key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_session() -> AsyncSession:
    """Dependency: yield an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def verify_api_key(
    api_key: str = Security(api_key_header),
) -> str:
    """Dependency: verify the API key from Authorization header."""
    settings = get_settings()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    # Support both "Bearer <key>" and plain "<key>" formats
    key = api_key
    if key.startswith("Bearer "):
        key = key[7:]

    if key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return key


async def verify_admin_key(
    api_key: str = Security(api_key_header),
) -> str:
    """Dependency: verify admin-level API key."""
    # For now, use the same key. Could be a separate admin key.
    return await verify_api_key(api_key)
