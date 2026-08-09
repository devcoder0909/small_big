"""Public routes — Ultra-minimal web UI and unauthenticated prediction readout."""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.services.analytics_service import get_prediction

router = APIRouter(tags=["public"])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WinGo 30S Prediction</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;900&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 26, 43, 0.85);
      --card-border: rgba(255, 255, 255, 0.1);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --small-color: #06b6d4;
      --small-glow: rgba(6, 182, 212, 0.45);
      --big-color: #f59e0b;
      --big-glow: rgba(245, 158, 11, 0.45);
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 30% 20%, rgba(6, 182, 212, 0.18) 0px, transparent 50%),
        radial-gradient(at 70% 80%, rgba(245, 158, 11, 0.18) 0px, transparent 50%);
      color: var(--text-main);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .card {
      width: 100%;
      max-width: 420px;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 28px;
      padding: 3rem 2rem;
      box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.6);
      text-align: center;
    }
    .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      font-weight: 700;
      color: #10b981;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      margin-bottom: 1.5rem;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 12px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    .label {
      font-size: 0.85rem;
      font-weight: 700;
      letter-spacing: 2px;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 0.75rem;
    }
    .result {
      font-size: 5rem;
      font-weight: 900;
      letter-spacing: 4px;
      line-height: 1;
      margin-bottom: 1.5rem;
      transition: all 0.4s ease;
    }
    .result.SMALL {
      color: var(--small-color);
      text-shadow: 0 0 40px var(--small-glow);
    }
    .result.BIG {
      color: var(--big-color);
      text-shadow: 0 0 40px var(--big-glow);
    }
    .issue {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      color: #cbd5e1;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 6px 16px;
      border-radius: 12px;
      display: inline-block;
      margin-bottom: 1.25rem;
    }
    .confidence {
      font-size: 0.95rem;
      font-weight: 700;
      color: #e2e8f0;
      background: rgba(255, 255, 255, 0.04);
      padding: 8px 20px;
      border-radius: 100px;
      display: inline-block;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="live-badge">
      <span class="pulse-dot"></span>
      <span>LIVE PREDICTION</span>
    </div>

    <div class="label">PREDICTED RESULT</div>

    <div id="result" class="result SMALL">LOADING</div>

    <div id="issue" class="issue">ISSUE #---</div>

    <div id="confidence" class="confidence">Confidence: --%</div>
  </div>

  <script>
    async function update() {
      try {
        const res = await fetch('/api/v1/public/prediction');
        const data = await res.json();
        
        if (data && data.prediction) {
          const resElem = document.getElementById('result');
          resElem.textContent = data.prediction;
          resElem.className = 'result ' + data.prediction;

          document.getElementById('issue').textContent = 'ISSUE #' + (data.upcoming_issue_id || '---');
          
          const confPct = (data.confidence * 100).toFixed(1);
          document.getElementById('confidence').textContent = `Confidence: ${confPct}% (${data.confidence_level || 'MED'})`;
        } else if (data && data.message) {
          document.getElementById('result').textContent = 'WAITING';
          document.getElementById('confidence').textContent = data.message;
        }
      } catch (err) {
        console.error('Fetch error:', err);
      }
    }
    update();
    setInterval(update, 3000);
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_minimal_ui():
    """Ultra-simple, minimal Web UI showing ONLY the predicted result."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """Public unauthenticated prediction readout for minimal UI."""
    return await get_prediction(session)
