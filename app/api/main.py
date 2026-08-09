"""FastAPI application — main entry point for the API server."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import get_settings
from app.core.logging import setup_logging
from app.api.routes import health, results, analytics, admin, public


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — setup and teardown."""
    setup_logging()
    try:
        from app.core.database import engine
        from app.models.base import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        from app.core.logging import get_logger
        get_logger(__name__).warning("db_schema_init_warning", error=str(e))
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="WinGo 30S Historical Data Platform",
        description=(
            "24/7 historical data collection and analytics platform for WinGo 30S. "
            "Provides verified historical results and statistical predictions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Data-Updated-At", "X-Api-Generated-At"],
    )

    # Include routers
    app.include_router(public.router)
    app.include_router(health.router)
    app.include_router(results.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/admin")

    return app


app = create_app()
