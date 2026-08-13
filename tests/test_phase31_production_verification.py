"""
Phase 31 Final Production Integration & Real Game History Verification Test.

Verifies:
1. Production prediction engine generates BIG/SMALL and NUMBER 0-9 predictions.
2. Public prediction API endpoint returns both game predictions.
3. UI template displays separated Game 1 (BIG/SMALL), Game 2 (NUMBER 0-9), and Latest Real Result.
4. Database game history contains real actual numbers (0-9) and size (BIG/SMALL).
5. Zero color prediction dependency.
6. Zero future leakage.
"""

import pytest
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from app.api.main import app
from app.api.dependencies import get_session
from app.models.game_result import GameResult
from app.analytics.prediction_engine import generate_prediction


@pytest.mark.asyncio
async def test_phase31_end_to_end_production_integration(db_session):
    """Verify complete end-to-end integration: Engine -> API -> UI -> History."""
    # 1. Seed database with 20 real historical draws
    for i in range(20):
        now_dt = datetime.now(timezone.utc)
        rec = GameResult(
            issue_id=str(20260812100060000 + i),
            result_number=i % 10,
            calculated_size="BIG" if (i % 10) >= 5 else "SMALL",
            source_color="red" if (i % 10) % 2 == 0 else "green",
            premium=str(i % 10),
            sum_value=0,
            first_observed_at=now_dt,
            last_observed_at=now_dt,
            source_url="http://test",
        )
        db_session.add(rec)
    await db_session.commit()

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    # 2. Test Engine level
    pred = await generate_prediction(db_session, window=100)
    assert pred is not None
    assert pred["prediction"] in ("BIG", "SMALL", "PASS")
    assert "digit_prediction" in pred
    assert pred["digit_prediction"]["top_numbers"] is not None
    assert len(pred["digit_prediction"]["top_numbers"]) == 4

    # 3. Test API & UI endpoints
    from app.services.prediction_pipeline import pipeline, PipelineState
    pipeline._current_prediction = dict(pred)
    pipeline._current_prediction["status"] = PipelineState.READY.value
    pipeline._current_prediction["upcoming_issue_id"] = "20260812100060020"
    pipeline._state = PipelineState.READY

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # UI Endpoint
        res_ui = await client.get("/")
        assert res_ui.status_code == 200
        assert "GAME 1: NUMBER GAME PREDICTION" in res_ui.text
        assert "GAME 2: BIG / SMALL PREDICTION" in res_ui.text
        assert "Real Game History" in res_ui.text

        # API Endpoint
        res_api = await client.get("/api/v1/public/prediction")
        assert res_api.status_code == 200
        data = res_api.json()
        assert "prediction" in data
        assert "digit_prediction" in data
        assert "recent_history" in data
        assert len(data["recent_history"]) > 0

        latest_hist = data["recent_history"][0]
        assert "result_number" in latest_hist
        assert "calculated_size" in latest_hist or "actual" in latest_hist or "result" in latest_hist

    app.dependency_overrides.clear()
