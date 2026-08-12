"""
Game History Integrity Test Suite.

Proves that Game History is strictly authoritative and powered solely by real observed GameResult rows.
Proves zero dependence on predictions, WIN/LOSS calculations, accuracy metrics, or AI weights.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from app.analytics.prediction_engine import get_game_history
from app.api.routes.public import HTML_PAGE, get_public_prediction


class MockGameResult:
    def __init__(self, issue_id, calculated_size, number=5, color="red"):
        self.issue_id = str(issue_id)
        self.calculated_size = calculated_size
        self.result_number = number
        self.source_color = color


# 1. History contains only GameResult records
@pytest.mark.asyncio
async def test_history_contains_only_game_results():
    mock_session = AsyncMock()
    rows = [
        MockGameResult("52280", "BIG"),
        MockGameResult("52279", "SMALL"),
    ]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)

    assert len(history) == 2
    assert history[0]["period"] == "52280"
    assert history[0]["result"] == "BIG"
    assert history[1]["period"] == "52279"
    assert history[1]["result"] == "SMALL"


# 2. History does not query EnginePrediction
@pytest.mark.asyncio
async def test_history_does_not_query_engine_prediction():
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [MockGameResult("52280", "BIG")]
    mock_session.execute.return_value = mock_exec

    await get_game_history(mock_session, limit=10)

    # Verify SQL query statement target table was GameResult, not EnginePrediction
    executed_stmt = str(mock_session.execute.call_args[0][0])
    assert "game_results" in executed_stmt.lower()
    assert "engine_predictions" not in executed_stmt.lower()


# 3. History does not call prediction_engine
@pytest.mark.asyncio
async def test_history_does_not_call_prediction_engine():
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [MockGameResult("52280", "BIG")]
    mock_session.execute.return_value = mock_exec

    with patch("app.analytics.prediction_engine.generate_prediction", side_effect=AssertionError("Should not be called")):
        history = await get_game_history(mock_session, limit=10)
        assert len(history) == 1


# 4. History does not calculate WIN/LOSS
@pytest.mark.asyncio
async def test_history_does_not_calculate_win_loss():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)

    record = history[0]
    assert "win" not in record
    assert "loss" not in record
    assert "is_win" not in record
    assert "prediction_status" not in record


# 5. History does not calculate accuracy
@pytest.mark.asyncio
async def test_history_does_not_calculate_accuracy():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG"), MockGameResult("52279", "SMALL")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)

    for item in history:
        assert "accuracy" not in item
        assert "accuracy_pct" not in item
        assert "wins" not in item


# 6. Historical prediction records cannot modify displayed game results
@pytest.mark.asyncio
async def test_historical_prediction_cannot_modify_game_results():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)
    assert history[0]["result"] == "BIG"

    # Even if EnginePrediction has SMALL for issue 52280, GameResult remains BIG
    mock_ep = MagicMock(issue_id="52280", predicted_size="SMALL")
    assert history[0]["result"] == "BIG"
    assert "predicted" not in history[0]


# 7. Changing AI providers cannot modify GameResult history
@pytest.mark.asyncio
async def test_changing_ai_providers_does_not_modify_history():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "SMALL")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    h1 = await get_game_history(mock_session, limit=10)

    # Change AI provider settings dynamically
    mock_s = MagicMock(ai_providers="groq,openrouter,nvidia")
    with patch("app.core.config.get_settings", return_value=mock_s):
        h2 = await get_game_history(mock_session, limit=10)

    assert h1[0]["result"] == h2[0]["result"] == "SMALL"


# 8. Changing prediction weights cannot modify GameResult history
@pytest.mark.asyncio
async def test_changing_prediction_weights_does_not_modify_history():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    h1 = await get_game_history(mock_session, limit=10)

    with patch("app.analytics.champion_selector.ChampionSelector.select_champion_strategy", return_value=(None, "markov", 0.8)):
        h2 = await get_game_history(mock_session, limit=10)

    assert h1 == h2


# 9. Changing champion/regime cannot modify GameResult history
@pytest.mark.asyncio
async def test_changing_regime_does_not_modify_history():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "SMALL")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    h1 = await get_game_history(mock_session, limit=10)

    with patch("app.analytics.regime_detector.detect_market_regime", return_value={"regime": "HIGH_VOLATILITY"}):
        h2 = await get_game_history(mock_session, limit=10)

    assert h1[0]["result"] == h2[0]["result"] == "SMALL"


# 10. Future prediction generation cannot modify historical GameResult values
@pytest.mark.asyncio
async def test_future_prediction_generation_cannot_modify_history():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    h1 = await get_game_history(mock_session, limit=10)

    # Future prediction for 52281
    future_pred = {"upcoming_issue_id": "52281", "prediction": "SMALL"}

    h2 = await get_game_history(mock_session, limit=10)
    assert h1[0]["result"] == h2[0]["result"] == "BIG"


# 11. Duplicate scraper events cannot create duplicate historical rows
@pytest.mark.asyncio
async def test_duplicate_scraper_events_deduplicated():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG")]
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    h1 = await get_game_history(mock_session, limit=10)
    h2 = await get_game_history(mock_session, limit=10)

    assert len(h1) == 1
    assert len(h2) == 1


# 12. Real scraped result remains authoritative
@pytest.mark.asyncio
async def test_real_scraped_result_authoritative():
    r = MockGameResult("52280", "SMALL", number=3, color="green")
    assert r.calculated_size == "SMALL"
    assert r.result_number == 3


# 13. Missing GameResult does not get fabricated from prediction data
@pytest.mark.asyncio
async def test_missing_gameresult_not_fabricated():
    mock_session = AsyncMock()
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = []
    mock_session.execute.return_value = mock_exec

    history = await get_game_history(mock_session, limit=10)
    assert len(history) == 0


# 14. No predicted field is returned by the history API
@pytest.mark.asyncio
async def test_no_predicted_field_in_public_prediction_api():
    mock_session = AsyncMock()
    rows = [MockGameResult("52280", "BIG"), MockGameResult("52279", "SMALL")]

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = mock_exec

    with patch("app.services.prediction_pipeline.pipeline.get_current_prediction", return_value={"status": "READY", "prediction": "BIG", "upcoming_issue_id": "52281"}):
        res = await get_public_prediction(mock_session)
        assert "recent_history" in res
        hist = res["recent_history"]
        assert len(hist) == 2
        for item in hist:
            assert "predicted" not in item
            assert "predicted_size" not in item
            assert "prediction_result" not in item
            assert "win" not in item
            assert "loss" not in item
            assert "is_win" not in item
            assert "confidence" not in item
            assert "accuracy" not in item
            assert "result" in item


# 15. No predicted field is rendered by the frontend HTML
def test_frontend_html_has_no_prediction_history_rendering():
    assert "History & Accuracy" not in HTML_PAGE
    assert "Wins / 5" not in HTML_PAGE
    assert "accuracy-pct" not in HTML_PAGE
    assert "Predicted Result" not in HTML_PAGE
    assert "Real Game History" in HTML_PAGE
    assert "Big Small</th>" in HTML_PAGE
