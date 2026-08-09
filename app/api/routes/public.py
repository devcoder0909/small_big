"""Public routes — Ultra-minimal zero-design prediction readout."""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.services.analytics_service import get_prediction

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
      height: 100vh;
      margin: 0;
      text-align: center;
    }
    #res {
      font-size: 10rem;
      font-weight: 900;
      line-height: 1;
    }
    #issue {
      font-size: 2rem;
      color: #888;
      margin-top: 1.5rem;
    }
    #conf {
      font-size: 1.5rem;
      color: #666;
      margin-top: 0.5rem;
    }
  </style>
</head>
<body>
  <div id="res">---</div>
  <div id="issue">#---</div>
  <div id="conf">--%</div>
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
    """Pure minimal UI focusing only on the result."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """Public unauthenticated prediction readout."""
    return await get_prediction(session)
