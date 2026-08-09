"""Tests for FastAPI API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.api.main import app
from app.api.dependencies import get_session


@pytest.mark.asyncio
async def test_health_endpoint(db_session):
    """Test public GET /health endpoint."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_unauthorized_access(db_session):
    """Test 401 response on protected endpoint without API key."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/latest")
        assert response.status_code == 401

    app.dependency_overrides.clear()
