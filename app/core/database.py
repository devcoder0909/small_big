"""Database engine and session management using SQLAlchemy 2.x async."""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import create_engine
from app.core import get_settings


def get_async_engine():
    """Create async database engine with connection pooling."""
    settings = get_settings()
    url = settings.database_url
    connect_args = {}

    # Handle SSL query parameters for asyncpg driver
    if "asyncpg" in url and ("ssl=" in url or "sslmode=" in url):
        # Convert sslmode query parameters to asyncpg connect_args
        if "sslmode=require" in url or "ssl=require" in url or "ssl=true" in url:
            connect_args["ssl"] = "require"
        url = (
            url.replace("?sslmode=require", "")
            .replace("&sslmode=require", "")
            .replace("?ssl=require", "")
            .replace("&ssl=require", "")
            .replace("?ssl=true", "")
            .replace("&ssl=true", "")
        )

    return create_async_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        echo=settings.app_env == "development",
    )


def get_sync_engine():
    """Create sync engine for Alembic migrations."""
    settings = get_settings()
    return create_engine(
        settings.database_sync_url,
        pool_pre_ping=True,
    )


# Global engine and session factory
engine = get_async_engine()
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session() -> AsyncSession:
    """Dependency: yield an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Check if database is reachable."""
    try:
        async with async_session_factory() as session:
            await session.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        return True
    except Exception:
        return False
