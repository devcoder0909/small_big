"""Tests for FastAPI API endpoints."""

import pytest
from unittest.mock import AsyncMock
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


@pytest.mark.asyncio
async def test_public_minimal_ui_and_prediction(db_session):
    """Test public UI endpoint / and public prediction API endpoint."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_ui = await client.get("/")
        assert res_ui.status_code == 200
        assert "<title>WinGo" in res_ui.text

        res_pred = await client.get("/api/v1/public/prediction")
        assert res_pred.status_code == 200
        data = res_pred.json()
        assert "status" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_public_prediction_500_resilience():
    """Regression test: verify /api/v1/public/prediction NEVER returns 500 on database error."""
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Database connection failure simulation")

    async def throwing_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = throwing_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_pred = await client.get("/api/v1/public/prediction")
        assert res_pred.status_code == 200
        data = res_pred.json()
        assert data["status"] == "INSUFFICIENT_DATA"
        assert "server_time_ms" in data

    app.dependency_overrides.clear()
