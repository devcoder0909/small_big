"""
Tests for health service and /health endpoint status classification.

Covers:
1. Healthy state (DB connected, Collector running/healthy).
2. Degraded state (DB connected, Collector stopped/unknown/stale).
3. Unhealthy state (DB disconnected).
"""

from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.system_heartbeat import SystemHeartbeat
from app.services.health_service import get_health, get_detailed_health


@pytest.mark.asyncio
async def test_get_health_healthy_state():
    """Verify healthy status when DB connected and collector healthy."""
    mock_session = AsyncMock()

    hb = SystemHeartbeat(
        service_name="collector",
        status="HEALTHY",
        last_successful_fetch=datetime.now(timezone.utc),
        last_new_record=datetime.now(timezone.utc),
        uptime_seconds=3600,
    )

    hb_res = MagicMock()
    hb_res.scalar_one_or_none.return_value = hb

    cnt_res = MagicMock()
    cnt_res.scalar.return_value = 1500

    mock_session.execute.side_effect = [hb_res, cnt_res]

    with patch("app.services.health_service.check_db_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        health = await get_health(mock_session)

        assert health["status"] == "healthy"
        assert health["collector"] == "healthy"
        assert health["database"] == "connected"
        assert health["records_total"] == 1500


@pytest.mark.asyncio
async def test_get_health_degraded_collector_stopped():
    """Verify degraded status when collector is stopped or unknown."""
    mock_session = AsyncMock()

    hb_res = MagicMock()
    hb_res.scalar_one_or_none.return_value = None

    cnt_res = MagicMock()
    cnt_res.scalar.return_value = 500

    mock_session.execute.side_effect = [hb_res, cnt_res]

    with patch("app.services.health_service.check_db_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        health = await get_health(mock_session)

        assert health["status"] == "degraded"
        assert health["collector"] == "unknown"
        assert health["database"] == "connected"


@pytest.mark.asyncio
async def test_get_health_degraded_stale_collector():
    """Verify degraded status when collector last_fetch is stale."""
    mock_session = AsyncMock()

    stale_fetch = datetime.now(timezone.utc) - timedelta(seconds=300)
    hb = SystemHeartbeat(
        service_name="collector",
        status="HEALTHY",
        last_successful_fetch=stale_fetch,
        last_new_record=stale_fetch,
        uptime_seconds=3600,
    )

    hb_res = MagicMock()
    hb_res.scalar_one_or_none.return_value = hb

    cnt_res = MagicMock()
    cnt_res.scalar.return_value = 800

    mock_session.execute.side_effect = [hb_res, cnt_res]

    with patch("app.services.health_service.check_db_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        health = await get_health(mock_session)

        assert health["status"] == "degraded"
        assert health["collector"] == "degraded"
        assert health["database"] == "connected"


@pytest.mark.asyncio
async def test_get_health_unhealthy_db_disconnected():
    """Verify unhealthy status when database is disconnected."""
    mock_session = AsyncMock()

    hb_res = MagicMock()
    hb_res.scalar_one_or_none.return_value = None

    cnt_res = MagicMock()
    cnt_res.scalar.return_value = 0

    mock_session.execute.side_effect = [hb_res, cnt_res]

    with patch("app.services.health_service.check_db_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False

        health = await get_health(mock_session)

        assert health["status"] == "unhealthy"
        assert health["database"] == "disconnected"


def test_get_build_commit_filters_unexpanded_placeholders():
    """Verify get_build_commit skips unexpanded template placeholders like '${NF_COMMIT_SHA}'."""
    from app.core.config import get_build_commit
    with patch("os.getenv") as mock_env:
        def fake_getenv(key, default=""):
            if key == "BUILD_COMMIT":
                return "${NF_COMMIT_SHA}"
            if key == "NF_COMMIT_SHA":
                return "3a64a1a"
            return default
        mock_env.side_effect = fake_getenv
        commit = get_build_commit()
        assert commit == "3a64a1a"

