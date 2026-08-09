"""Public routes — Premium mobile-first prediction UI with live countdown timer and accuracy tracker."""

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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>WinGo Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0a0f;
      --card: #12121a;
      --border: #1e1e2e;
      --text: #e4e4ef;
      --muted: #6b6b80;
      --accent-big: #ff4d6a;
      --accent-small: #4da6ff;
      --green: #00d68f;
      --red: #ff4d6a;
      --gold: #ffd700;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      min-height: 100dvh;
      padding: 0.75rem;
      -webkit-font-smoothing: antialiased;
    }
    .container {
      max-width: 420px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      padding-bottom: 2rem;
    }

    /* Header */
    .header {
      text-align: center;
      padding: 0.6rem 0 0.2rem;
    }
    .header h1 {
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: 3px;
      text-transform: uppercase;
      background: linear-gradient(135deg, var(--accent-small), var(--accent-big));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .header .sub {
      font-size: 0.7rem;
      color: var(--muted);
      margin-top: 0.15rem;
      letter-spacing: 1px;
    }

    /* Timer Card */
    .timer-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      text-align: center;
    }
    .timer-label {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--muted);
    }
    .timer-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: 2.8rem;
      font-weight: 700;
      color: var(--gold);
      line-height: 1.1;
      margin: 0.4rem 0;
    }
    .timer-bar-bg {
      width: 100%;
      height: 4px;
      background: #1a1a2e;
      border-radius: 2px;
      overflow: hidden;
      margin-top: 0.5rem;
    }
    .timer-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--gold), #ff9500);
      border-radius: 2px;
      transition: width 0.3s linear;
    }
    .timer-period {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.4rem;
    }

    /* Prediction Card */
    .pred-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem 1rem;
      text-align: center;
      position: relative;
      overflow: hidden;
    }
    .pred-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--accent-small), var(--accent-big));
    }
    .pred-label {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--muted);
      margin-bottom: 0.6rem;
    }
    .pred-signal {
      font-size: 4.5rem;
      font-weight: 900;
      line-height: 1;
      letter-spacing: -2px;
      transition: all 0.3s ease;
    }
    .pred-signal.big { color: var(--accent-big); }
    .pred-signal.small { color: var(--accent-small); }
    .pred-signal.waiting { color: var(--muted); font-size: 2.5rem; }
    .pred-conf {
      margin-top: 0.6rem;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .pred-conf .conf-value {
      font-weight: 700;
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
    }
    .conf-badge {
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-left: 0.4rem;
    }
    .conf-high { background: rgba(0,214,143,0.15); color: var(--green); border: 1px solid rgba(0,214,143,0.3); }
    .conf-med { background: rgba(255,215,0,0.12); color: var(--gold); border: 1px solid rgba(255,215,0,0.25); }
    .conf-low { background: rgba(107,107,128,0.15); color: var(--muted); border: 1px solid rgba(107,107,128,0.25); }

    /* Stats Row */
    .stats-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 0.5rem;
    }
    .stat-box {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.7rem 0.5rem;
      text-align: center;
    }
    .stat-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
    }
    .stat-val.green { color: var(--green); }
    .stat-val.gold { color: var(--gold); }
    .stat-val.blue { color: var(--accent-small); }
    .stat-lbl {
      font-size: 0.6rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 0.2rem;
    }

    /* History */
    .history-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
    }
    .history-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }
    .history-title {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--muted);
    }
    .accuracy-pct {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--green);
    }
    .history-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.6rem 0;
      border-bottom: 1px solid rgba(30,30,46,0.6);
    }
    .history-row:last-child { border-bottom: none; }
    .history-left { display: flex; align-items: center; gap: 0.6rem; }
    .history-num {
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 0.85rem;
    }
    .history-num.big-bg { background: rgba(255,77,106,0.15); color: var(--accent-big); }
    .history-num.small-bg { background: rgba(77,166,255,0.15); color: var(--accent-small); }
    .history-info { display: flex; flex-direction: column; }
    .history-issue {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
    }
    .history-size {
      font-size: 0.8rem;
      font-weight: 700;
    }
    .history-size.big { color: var(--accent-big); }
    .history-size.small { color: var(--accent-small); }
    .badge {
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge-win { background: rgba(0,214,143,0.12); color: var(--green); border: 1px solid rgba(0,214,143,0.25); }
    .badge-loss { background: rgba(255,77,106,0.12); color: var(--red); border: 1px solid rgba(255,77,106,0.25); }

    /* Disclaimer */
    .disclaimer {
      text-align: center;
      font-size: 0.6rem;
      color: #444;
      line-height: 1.5;
      padding: 0.5rem 1rem;
    }

    /* Pulse animation for timer urgency */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    .urgent { animation: pulse 0.6s ease-in-out infinite; color: var(--red) !important; }

    /* Responsive */
    @media (max-width: 380px) {
      .pred-signal { font-size: 3.5rem; }
      .timer-display { font-size: 2.2rem; }
      .stat-val { font-size: 0.95rem; }
    }
    @media (min-width: 500px) {
      body { padding: 1.5rem; }
      .container { gap: 1rem; }
      .pred-signal { font-size: 5.5rem; }
      .timer-display { font-size: 3.2rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>WinGo Predictor</h1>
      <div class="sub">10-Indicator Statistical Ensemble Engine</div>
    </div>

    <div class="timer-card">
      <div class="timer-label">Next Round Closes In</div>
      <div class="timer-display" id="timer">--</div>
      <div class="timer-bar-bg"><div class="timer-bar-fill" id="timer-bar" style="width:100%"></div></div>
      <div class="timer-period" id="timer-period">Period #---</div>
    </div>

    <div class="pred-card">
      <div class="pred-label">Next Draw Prediction</div>
      <div class="pred-signal waiting" id="pred-signal">---</div>
      <div class="pred-conf" id="pred-conf">Confidence: --%</div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-val green" id="stat-wins">-</div>
        <div class="stat-lbl">Wins / 5</div>
      </div>
      <div class="stat-box">
        <div class="stat-val gold" id="stat-indicators">-</div>
        <div class="stat-lbl">Indicators</div>
      </div>
      <div class="stat-box">
        <div class="stat-val blue" id="stat-records">-</div>
        <div class="stat-lbl">Records</div>
      </div>
    </div>

    <div class="history-card">
      <div class="history-header">
        <span class="history-title">Recent Draws & Accuracy</span>
        <span class="accuracy-pct" id="accuracy-pct">--%</span>
      </div>
      <div id="history-list">
        <div class="history-row" style="justify-content:center;color:var(--muted);font-size:0.8rem;">Loading...</div>
      </div>
    </div>

    <div class="disclaimer">
      Statistical analysis based on historical patterns.<br>
      Each draw is an independent random event. Not a guarantee.
    </div>
  </div>

  <script>
    // WinGo 30S countdown timer
    function updateTimer() {
      const now = new Date();
      const seconds = now.getSeconds();
      const remaining = 30 - (seconds % 30);
      const pct = (remaining / 30) * 100;

      const timerEl = document.getElementById('timer');
      const barEl = document.getElementById('timer-bar');

      timerEl.textContent = '00:' + String(remaining).padStart(2, '0');
      barEl.style.width = pct + '%';

      if (remaining <= 5) {
        timerEl.classList.add('urgent');
        barEl.style.background = 'linear-gradient(90deg, var(--red), #ff6b6b)';
      } else {
        timerEl.classList.remove('urgent');
        barEl.style.background = 'linear-gradient(90deg, var(--gold), #ff9500)';
      }
    }

    async function update() {
      try {
        const r = await fetch('/api/v1/public/prediction');
        const d = await r.json();

        const sigEl = document.getElementById('pred-signal');
        const confEl = document.getElementById('pred-conf');
        const periodEl = document.getElementById('timer-period');

        if (d && d.prediction) {
          sigEl.textContent = d.prediction;
          sigEl.className = 'pred-signal ' + d.prediction.toLowerCase();
          periodEl.textContent = 'Period #' + (d.upcoming_issue_id || '').slice(-8);

          const confPct = (d.confidence * 100).toFixed(1);
          const level = d.confidence >= 0.75 ? 'HIGH' : d.confidence >= 0.55 ? 'MED' : 'LOW';
          const levelClass = d.confidence >= 0.75 ? 'conf-high' : d.confidence >= 0.55 ? 'conf-med' : 'conf-low';
          confEl.innerHTML = 'Confidence: <span class="conf-value">' + confPct + '%</span> <span class="conf-badge ' + levelClass + '">' + level + '</span>';

          // Stats
          document.getElementById('stat-indicators').textContent = (d.active_indicators || '-') + '/' + (d.agreeing_indicators || '-');
          document.getElementById('stat-records').textContent = d.total_records_analyzed || '-';
        } else if (d && d.status === 'INSUFFICIENT_DATA') {
          sigEl.textContent = 'WAIT';
          sigEl.className = 'pred-signal waiting';
          periodEl.textContent = 'Collecting data...';
          confEl.innerHTML = 'Initializing engine...';
        }

        if (d && d.recent_history && d.recent_history.length > 0) {
          const hist = d.recent_history;
          const wins = hist.filter(h => h.is_win).length;
          const total = hist.length;
          const pct = total > 0 ? Math.round((wins / total) * 100) : 0;

          document.getElementById('stat-wins').textContent = wins + '/' + total;
          document.getElementById('accuracy-pct').textContent = pct + '% Accurate';
          document.getElementById('accuracy-pct').style.color = pct >= 60 ? 'var(--green)' : pct >= 40 ? 'var(--gold)' : 'var(--red)';

          const listEl = document.getElementById('history-list');
          listEl.innerHTML = hist.map(item => {
            const isBig = item.size === 'BIG';
            return '<div class="history-row">' +
              '<div class="history-left">' +
                '<div class="history-num ' + (isBig ? 'big-bg' : 'small-bg') + '">' + item.result + '</div>' +
                '<div class="history-info">' +
                  '<span class="history-issue">#' + item.issue_id.slice(-8) + '</span>' +
                  '<span class="history-size ' + (isBig ? 'big' : 'small') + '">' + item.size + '</span>' +
                '</div>' +
              '</div>' +
              '<span class="badge ' + (item.is_win ? 'badge-win' : 'badge-loss') + '">' +
                (item.is_win ? '✅ WIN' : '❌ LOSS') +
              '</span>' +
            '</div>';
          }).join('');
        }
      } catch (e) {}
    }

    updateTimer();
    update();
    setInterval(updateTimer, 250);
    setInterval(update, 2000);
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_minimal_ui():
    """Premium mobile-first prediction UI with live countdown and accuracy tracker."""
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
