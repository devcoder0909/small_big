"""Public routes — Ultra-minimal zero-design prediction readout with real game history and prediction verification."""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.models.game_result import GameResult
from app.services.analytics_service import get_prediction
from app.analytics.prediction_engine import evaluate_recent_accuracy

router = APIRouter(tags=["public"])

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prediction</title>
  <style>
    body {
      background: #000;
      color: #fff;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 2rem 1rem;
      box-sizing: border-box;
      text-align: center;
    }
    .prediction-card {
      background: #0a0a0a;
      border: 1px solid #222;
      border-radius: 16px;
      padding: 2.5rem 2rem;
      width: 100%;
      max-width: 480px;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    }
    .section-label {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #777;
      margin-bottom: 0.8rem;
    }
    #res {
      font-size: 6.5rem;
      font-weight: 900;
      line-height: 1;
      letter-spacing: -2px;
      margin: 0.5rem 0;
    }
    .tag-big-res { color: #ff4d4d; }
    .tag-small-res { color: #4da6ff; }
    #issue {
      font-size: 1.2rem;
      color: #aaa;
      font-family: monospace;
      margin-top: 0.8rem;
    }
    #conf {
      font-size: 1.1rem;
      color: #888;
      margin-top: 0.4rem;
    }
    #history-container {
      margin-top: 2.5rem;
      width: 100%;
      max-width: 480px;
      text-align: left;
    }
    .history-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #777;
      margin-bottom: 0.8rem;
      border-bottom: 1px solid #222;
      padding-bottom: 0.5rem;
    }
    .history-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.8rem 0;
      border-bottom: 1px solid #111;
      font-size: 0.95rem;
    }
    .history-left {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .history-issue {
      color: #888;
      font-family: monospace;
      font-size: 0.85rem;
    }
    .history-draw {
      font-weight: bold;
      color: #eee;
    }
    .tag-big { color: #ff4d4d; font-weight: bold; }
    .tag-small { color: #4da6ff; font-weight: bold; }
    .badge-win {
      background: rgba(46, 204, 113, 0.15);
      color: #2ecc71;
      border: 1px solid rgba(46, 204, 113, 0.3);
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
    }
    .badge-loss {
      background: rgba(231, 76, 60, 0.15);
      color: #e74c3c;
      border: 1px solid rgba(231, 76, 60, 0.3);
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <div class="prediction-card">
    <div class="section-label">NEXT DRAW PREDICTION</div>
    <div id="res">---</div>
    <div id="issue">Period #---</div>
    <div id="conf">Confidence: --%</div>
  </div>

  <div id="history-container">
    <div class="history-title">LAST 5 REAL DRAWS & PREDICTION ACCURACY</div>
    <div id="history-list">
      <div class="history-row" style="color: #444;">Loading real game history...</div>
    </div>
  </div>

  <script>
    async function update() {
      try {
        const r = await fetch('/api/v1/public/prediction');
        const d = await r.json();
        const resEl = document.getElementById('res');

        if (d && d.prediction) {
          resEl.textContent = d.prediction;
          resEl.className = d.prediction === 'BIG' ? 'tag-big-res' : 'tag-small-res';
          document.getElementById('issue').textContent = 'Period #' + (d.upcoming_issue_id || '');
          document.getElementById('conf').textContent = 'Prediction Confidence: ' + (d.confidence * 100).toFixed(1) + '%';
        } else if (d && d.status === 'INSUFFICIENT_DATA') {
          resEl.textContent = 'WAITING';
          resEl.className = '';
          document.getElementById('issue').textContent = 'Period #INITIALIZING';
          document.getElementById('conf').textContent = 'Collecting History Data...';
        }

        if (d && d.recent_history && d.recent_history.length > 0) {
          const listEl = document.getElementById('history-list');
          listEl.innerHTML = d.recent_history.map(item => `
            <div class="history-row">
              <div class="history-left">
                <span class="history-issue">#${item.issue_id.slice(-8)}</span>
                <span class="history-draw">
                  Num: ${item.result} &nbsp;(<span class="${item.size === 'BIG' ? 'tag-big' : 'tag-small'}">${item.size}</span>)
                </span>
              </div>
              <div class="history-right">
                <span class="${item.is_win ? 'badge-win' : 'badge-loss'}">
                  ${item.is_win ? '✅ ACCURATE WIN' : '❌ LOSS'}
                </span>
              </div>
            </div>
          `).join('');
        }
      } catch (e) {}
    }
    update();
    setInterval(update, 2000);
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_minimal_ui():
    """Pure minimal UI focusing on prediction and last 5 real draw accuracy."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """Public unauthenticated prediction readout with recent accuracy verification."""
    prediction = await get_prediction(session)

    # Non-blocking on-demand backfill if database is empty
    if prediction.get("status") == "INSUFFICIENT_DATA":
        import asyncio
        async def _bg_seed():
            try:
                from app.core.database import async_session_factory
                from app.services.recovery_service import recover_missing_records
                async with async_session_factory() as s:
                    async with s.begin():
                        await recover_missing_records(s)
            except Exception:
                pass
        asyncio.create_task(_bg_seed())

    # Fetch recent rows to evaluate accuracy
    rows_query = await session.execute(
        select(GameResult).order_by(desc(GameResult.issue_id)).limit(20)
    )
    rows = rows_query.scalars().all()
    prediction["recent_history"] = evaluate_recent_accuracy(rows)
    return prediction
