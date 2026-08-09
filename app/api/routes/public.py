"""Public routes — Event-driven prediction UI with ANALYZING → READY state machine."""

import time
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.models.game_result import GameResult
from app.analytics.prediction_engine import evaluate_recent_accuracy

router = APIRouter(tags=["public"])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>WinGo Predictor</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#0d0d12;color:#eee;font-family:system-ui,sans-serif;padding:12px;max-width:400px;margin:0 auto}
    h1{font-size:15px;text-align:center;color:#aaa;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
    .box{background:#14141c;border:1px solid #222;border-radius:6px;padding:12px;margin-bottom:10px}
    .lbl{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
    .sub{font-size:11px;color:#666;margin-top:2px}
    .pred-val{font-size:44px;font-weight:900;margin:4px 0;line-height:1}
    .pred-val.big{color:#ff4d6a}
    .pred-val.small{color:#4da6ff}
    .pred-val.wait{color:#555;font-size:24px}
    .pred-val.analyzing{color:#ffd700;font-size:20px}
    .conf-txt{font-size:12px;color:#888}
    .conf-txt b{color:#fff}
    .badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700}
    .badge.high{background:#00d68f22;color:#00d68f}
    .badge.med{background:#ffd70022;color:#ffd700}
    .badge.low{background:#88888822;color:#aaa}
    .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px}
    .g-box{background:#14141c;border:1px solid #222;border-radius:6px;padding:8px;text-align:center}
    .g-val{font-size:15px;font-weight:700;font-family:monospace}
    .g-val.green{color:#00d68f}.g-val.gold{color:#ffd700}.g-val.blue{color:#4da6ff}
    .g-lbl{font-size:9px;color:#666;text-transform:uppercase;margin-top:2px}
    table{width:100%;border-collapse:collapse;font-size:12px}
    th{text-align:left;font-size:10px;color:#666;padding-bottom:6px;text-transform:uppercase;border-bottom:1px solid #222}
    td{padding:6px 0;border-bottom:1px solid #1a1a24;font-family:monospace}
    .tag{display:inline-block;padding:1px 5px;border-radius:3px;font-size:11px;font-weight:700}
    .tag.big{color:#ff4d6a}.tag.small{color:#4da6ff}
    .win{color:#00d68f;font-weight:700}
    .loss{color:#ff4d6a;font-weight:700}
    .foot{text-align:center;font-size:9px;color:#444;margin-top:8px}
    .status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px}
    .status-dot.ready{background:#00d68f}
    .status-dot.analyzing{background:#ffd700}
    .status-dot.waiting{background:#555}
  </style>
</head>
<body>
  <h1>WinGo Predictor Engine</h1>

  <div class="box" style="text-align:center">
    <div class="lbl" id="pred-label" style="font-size:13px;font-weight:700;color:#aaa;margin-bottom:6px">PERIOD #---</div>
    <div class="pred-val wait" id="pred-text">---</div>
    <div class="conf-txt" id="conf-text">Confidence: <b>--%</b> — <span class="badge low">LOW</span></div>
    <div class="sub" id="pred-status" style="margin-top:6px;font-size:10px;color:#555"></div>
  </div>

  <div class="grid">
    <div class="g-box"><div class="g-val green" id="stat-wins">-</div><div class="g-lbl">Wins / 5</div></div>
    <div class="g-box"><div class="g-val gold" id="stat-signals">-</div><div class="g-lbl">Active Signals</div></div>
    <div class="g-box"><div class="g-val blue" id="stat-records">-</div><div class="g-lbl">Records</div></div>
  </div>

  <div class="box">
    <div class="lbl" style="margin-bottom:8px;display:flex;justify-content:space-between">
      <span>History & Accuracy</span>
      <span id="accuracy-pct" style="color:#00d68f;font-weight:700">--%</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Period</th>
          <th>Actual</th>
          <th>Predicted</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody id="history-body">
        <tr><td colspan="4" style="text-align:center;color:#444;padding:12px">Loading history...</td></tr>
      </tbody>
    </table>
  </div>

  <div class="foot">Event-Driven Pipeline. Auto-Analyzed on New Result.</div>

<script>
var lastRenderedKey = "";
var currentPredictionData = null;

async function updateData() {
  try {
    var res = await fetch('/api/v1/public/prediction');
    var data = await res.json();

    if (!data) return;

    var status = data.status || "";
    var issueId = data.upcoming_issue_id || "";
    var periodDisplay = issueId ? issueId.slice(-8) : "---";

    // ANALYZING state — new period, prediction not ready yet
    if (status === "ANALYZING") {
      var analyzeKey = "analyzing_" + periodDisplay;
      if (analyzeKey !== lastRenderedKey) {
        lastRenderedKey = analyzeKey;
        document.getElementById('pred-label').textContent = 'PERIOD #' + periodDisplay;
        var predEl = document.getElementById('pred-text');
        predEl.textContent = 'ANALYZING...';
        predEl.className = 'pred-val analyzing';
        document.getElementById('conf-text').innerHTML = 'Processing latest result...';
        document.getElementById('pred-status').innerHTML = '<span class="status-dot analyzing"></span>Generating prediction';
        document.getElementById('stat-signals').textContent = '-';
        document.getElementById('stat-records').textContent = '-';
      }
    }
    // READY / ACTIVE state — prediction locked and available
    else if (data.prediction && (status === "READY" || status === "ACTIVE")) {
      var renderKey = periodDisplay + '_' + data.prediction + '_' + (data.confidence || 0);
      if (renderKey !== lastRenderedKey) {
        lastRenderedKey = renderKey;
        currentPredictionData = data;

        document.getElementById('pred-label').textContent = 'PERIOD #' + periodDisplay;

        var predEl = document.getElementById('pred-text');
        predEl.textContent = data.prediction;
        predEl.className = 'pred-val ' + data.prediction.toLowerCase();

        var confPct = (data.confidence * 100).toFixed(1);
        var isSuper = data.confluence_level === 'SUPER_CONFLUENCE';
        var level = isSuper ? 'SUPER CONFLUENCE' : (data.confidence >= 0.72 ? 'HIGH' : data.confidence >= 0.56 ? 'MED' : 'LOW');
        var badgeClass = (isSuper || data.confidence >= 0.72) ? 'high' : data.confidence >= 0.56 ? 'med' : 'low';

        document.getElementById('conf-text').innerHTML = 'Confidence: <b>' + confPct + '%</b> — <span class="badge ' + badgeClass + '">' + level + '</span>';
        document.getElementById('pred-status').innerHTML = '<span class="status-dot ready"></span>Prediction Ready';
        document.getElementById('stat-signals').textContent = (data.agreeing_indicators || '-') + '/' + (data.active_indicators || '-');
        document.getElementById('stat-records').textContent = data.total_records_analyzed || '-';
      }
    }
    // INSUFFICIENT_DATA state
    else if (status === "INSUFFICIENT_DATA") {
      document.getElementById('pred-label').textContent = 'PERIOD #---';
      document.getElementById('pred-text').textContent = 'WAIT';
      document.getElementById('pred-text').className = 'pred-val wait';
      document.getElementById('conf-text').textContent = 'Collecting historical records...';
      document.getElementById('pred-status').innerHTML = '<span class="status-dot waiting"></span>Waiting for data';
    }

    // History section
    if (data.recent_history && data.recent_history.length > 0) {
      var hist = data.recent_history;
      var wins = hist.filter(function(x) { return x.is_win; }).length;
      var total = hist.length;
      var pct = Math.round((wins / total) * 100);

      document.getElementById('stat-wins').textContent = wins + '/' + total;
      var accEl = document.getElementById('accuracy-pct');
      accEl.textContent = pct + '%';
      accEl.style.color = pct >= 60 ? '#00d68f' : pct >= 40 ? '#ffd700' : '#ff4d6a';

      var tbody = document.getElementById('history-body');
      tbody.innerHTML = hist.map(function(item) {
        var actClass = item.size === 'BIG' ? 'big' : 'small';
        var predClass = item.predicted_size === 'BIG' ? 'big' : 'small';
        var resClass = item.is_win ? 'win' : 'loss';
        var resText = item.is_win ? '✅ WIN' : '❌ LOSS';

        return '<tr>' +
          '<td>#' + item.issue_id.slice(-8) + '</td>' +
          '<td><span class="tag ' + actClass + '">' + item.size + '</span></td>' +
          '<td><span class="tag ' + predClass + '">' + item.predicted_size + '</span></td>' +
          '<td class="' + resClass + '">' + resText + '</td>' +
        '</tr>';
      }).join('');
    }
  } catch (e) {}
}

updateData();
setInterval(updateData, 1500);
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_minimal_ui():
    """Event-driven prediction UI with ANALYZING → READY state machine."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """
    Public prediction endpoint — reads from the event-driven pipeline.

    The pipeline pre-computes predictions immediately after each new result is committed.
    This endpoint is a fast cache reader, not an on-demand generator.
    """
    try:
        from app.services.prediction_pipeline import pipeline

        prediction = pipeline.get_current_prediction()

        # If pipeline has no prediction yet (first startup), force an initial generation
        if prediction.get("status") == "INSUFFICIENT_DATA" and not prediction.get("upcoming_issue_id"):
            await pipeline.force_refresh()
            prediction = pipeline.get_current_prediction()

        prediction["server_time_ms"] = int(time.time() * 1000)

        # Attach recent accuracy history from immutable audit trail
        try:
            rows_query = await session.execute(
                select(GameResult).order_by(desc(GameResult.issue_id)).limit(20)
            )
            rows = rows_query.scalars().all()
            prediction["recent_history"] = await evaluate_recent_accuracy(session, rows)
        except Exception as err:
            from app.core.logging import get_logger
            get_logger(__name__).warning("evaluate_recent_accuracy_route_error", error=str(err))
            prediction["recent_history"] = []

        return prediction
    except Exception as exc:
        from app.core.logging import get_logger
        get_logger(__name__).error("public_prediction_endpoint_error", error=str(exc))
        return {
            "status": "INSUFFICIENT_DATA",
            "prediction": None,
            "confidence": 0,
            "message": "System initializing or recovering data",
            "server_time_ms": int(time.time() * 1000),
            "recent_history": [],
        }
