"""Public routes — Ultra-minimal zero-design prediction readout with real game history."""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.services.analytics_service import get_prediction
from app.services.result_service import get_results

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
    #res {
      font-size: 8rem;
      font-weight: 900;
      line-height: 1;
      letter-spacing: -2px;
    }
    #issue {
      font-size: 1.8rem;
      color: #888;
      margin-top: 1rem;
    }
    #conf {
      font-size: 1.4rem;
      color: #666;
      margin-top: 0.4rem;
    }
    #history-container {
      margin-top: 3rem;
      width: 100%;
      max-width: 480px;
      text-align: left;
    }
    .history-title {
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #555;
      margin-bottom: 0.8rem;
      border-bottom: 1px solid #222;
      padding-bottom: 0.4rem;
    }
    .history-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 0;
      border-bottom: 1px solid #111;
      font-size: 1rem;
    }
    .history-issue {
      color: #aaa;
      font-family: monospace;
    }
    .history-num {
      font-weight: bold;
      color: #fff;
    }
    .tag-big {
      color: #ff4d4d;
      font-weight: bold;
    }
    .tag-small {
      color: #4da6ff;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div id="res">---</div>
  <div id="issue">#---</div>
  <div id="conf">--%</div>

  <div id="history-container">
    <div class="history-title">LAST 5 REAL GAME DRAWS</div>
    <div id="history-list">
      <div class="history-row" style="color: #444;">Fetching real game history...</div>
    </div>
  </div>

  <script>
    async function update() {
      try {
        const r = await fetch('/api/v1/public/prediction');
        const d = await r.json();
        if (d && d.prediction) {
          document.getElementById('res').textContent = d.prediction;
          document.getElementById('issue').textContent = '#' + (d.upcoming_issue_id || '');
          document.getElementById('conf').textContent = (d.confidence * 100).toFixed(1) + '%';
        } else if (d && d.status === 'INSUFFICIENT_DATA') {
          document.getElementById('res').textContent = 'WAITING';
          document.getElementById('issue').textContent = '#INITIALIZING';
          document.getElementById('conf').textContent = 'Collecting History Data...';
        }

        if (d && d.recent_history && d.recent_history.length > 0) {
          const listEl = document.getElementById('history-list');
          listEl.innerHTML = d.recent_history.map(item => `
            <div class="history-row">
              <span class="history-issue">#${item.issue_id.slice(-8)}</span>
              <span class="history-num">Num: ${item.result}</span>
              <span class="${item.size === 'BIG' ? 'tag-big' : 'tag-small'}">${item.size}</span>
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
    """Pure minimal UI focusing on the result and last 5 real game draw history."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """Public unauthenticated prediction readout with last 5 real game draws."""
    prediction = await get_prediction(session)
    recent = await get_results(session, limit=5)
    prediction["recent_history"] = recent.get("results", [])
    return prediction
