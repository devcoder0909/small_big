"""Public routes — Event-driven prediction UI with ANALYZING → READY state machine."""

import time
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.models.game_result import GameResult
from app.analytics.prediction_engine import get_game_history

router = APIRouter(tags=["public"])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>WinGo Dual Prediction Engine — BIG/SMALL & NUMBER GAME</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#0d0d12;color:#eee;font-family:system-ui,sans-serif;padding:12px;max-width:420px;margin:0 auto}
    h1{font-size:15px;text-align:center;color:#aaa;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
    .period-title{font-size:13px;font-weight:700;color:#ffd700;text-align:center;margin-bottom:8px;letter-spacing:1px}
    .box{background:#14141c;border:1px solid #222;border-radius:8px;padding:14px;margin-bottom:10px}
    .lbl{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;font-weight:700}
    .sub{font-size:11px;color:#777;margin-top:4px}
    .pred-val{font-size:42px;font-weight:900;margin:2px 0;line-height:1}
    .pred-val.big{color:#ff4d6a}
    .pred-val.small{color:#4da6ff}
    .pred-val.wait{color:#555;font-size:24px}
    .pred-val.analyzing{color:#ffd700;font-size:20px}
    .digit-val{font-size:38px;font-weight:900;color:#ffd700;margin:2px 0;line-height:1}
    .conf-txt{font-size:12px;color:#aaa}
    .conf-txt b{color:#fff}
    .badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700}
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
    .foot{text-align:center;font-size:9px;color:#444;margin-top:8px}
    .status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px}
    .status-dot.ready{background:#00d68f}
    .status-dot.analyzing{background:#ffd700}
    .status-dot.waiting{background:#555}
  </style>
</head>
<body>
  <h1>WinGo Predictor Engine</h1>
  <div class="period-title" id="pred-label">PERIOD #---</div>

  <!-- GAME 1: NUMBER GAME -->
  <div class="box" style="text-align:center" id="digit-card">
    <div class="lbl">GAME 1: NUMBER GAME PREDICTION</div>
    <div class="digit-val" id="primary-digit">-</div>
    <div class="lbl" style="font-size:9px;color:#666;margin-top:4px">TOP 4 PROJECTED NUMBERS</div>
    <div id="digit-pills" style="font-size:22px;font-weight:900;color:#00d68f;letter-spacing:4px;margin:2px 0">---</div>
    <div class="sub" id="digit-sub" style="font-size:11px;color:#888">Top-4 Coverage: --%</div>
  </div>

  <!-- GAME 2: BIG / SMALL -->
  <div class="box" style="text-align:center">
    <div class="lbl">GAME 2: BIG / SMALL PREDICTION</div>
    <div class="pred-val wait" id="pred-text">---</div>
    <div class="conf-txt" id="conf-text">Confidence: <b>--%</b> — <span class="badge low">LOW</span></div>
    <div class="sub" id="pred-status" style="margin-top:4px;font-size:10px;color:#555"></div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
    <div class="g-box"><div class="g-val gold" id="stat-signals">-</div><div class="g-lbl">Active Signals</div></div>
    <div class="g-box"><div class="g-val blue" id="stat-records">-</div><div class="g-lbl">Historical Records</div></div>
    <div class="g-box"><div class="g-val green" id="stat-window">-</div><div class="g-lbl">Active Window</div></div>
  </div>

  <!-- LATEST COMPLETED REAL GAME RESULT -->
  <div class="box" style="text-align:center;border-color:#333;background:#181824" id="latest-result-card">
    <div class="lbl" style="color:#ffd700">LATEST REAL COMPLETED RESULT</div>
    <div id="latest-period-label" style="font-size:11px;color:#aaa;margin-bottom:2px">PERIOD #---</div>
    <div style="display:flex;justify-content:center;align-items:center;gap:16px;margin:6px 0;flex-wrap:wrap">
      <div><span style="font-size:10px;color:#888;text-transform:uppercase">Number:</span> <b style="font-size:24px;font-weight:900;color:#ffd700" id="latest-num-val">-</b></div>
      <div><span style="font-size:10px;color:#888;text-transform:uppercase">Big Small:</span> <span class="tag" style="font-size:14px;padding:3px 8px" id="latest-size-val">---</span></div>
      <div><span style="font-size:10px;color:#888;text-transform:uppercase">Color:</span> <span id="latest-color-val" style="font-size:13px;font-weight:bold;color:#ccc">---</span></div>
    </div>
    <div class="sub" style="font-size:9px;color:#666">Verified Real Game Result Record</div>
  </div>

  <div class="box">
    <div class="lbl" style="margin-bottom:8px">
      <span>Real Game History (Actual Number, Size & Color)</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Period</th>
          <th>Number</th>
          <th>Big Small</th>
          <th style="text-align:right">Color</th>
        </tr>
      </thead>
      <tbody id="history-body">
        <tr><td colspan="4" style="text-align:center;color:#444;padding:12px">Loading history...</td></tr>
      </tbody>
    </table>
  </div>

  <div class="foot">Event-Driven Engine • Auto-Analyzed on New Result</div>

<script>
var lastRenderedKey = "";
var currentPredictionData = null;

function formatColorHtml(colorStr) {
  if (!colorStr) return '<span style="color:#666">-</span>';
  var c = String(colorStr).toLowerCase();
  if (c.indexOf('red') !== -1 && c.indexOf('violet') !== -1) {
    return '<span style="color:#ff4d4d">●</span><span style="color:#b388ff">● Red/Violet</span>';
  } else if (c.indexOf('green') !== -1 && c.indexOf('violet') !== -1) {
    return '<span style="color:#00e676">●</span><span style="color:#b388ff">● Green/Violet</span>';
  } else if (c.indexOf('red') !== -1) {
    return '<span style="color:#ff4d4d">● Red</span>';
  } else if (c.indexOf('green') !== -1) {
    return '<span style="color:#00e676">● Green</span>';
  } else if (c.indexOf('violet') !== -1) {
    return '<span style="color:#b388ff">● Violet</span>';
  }
  return '<span style="color:#ccc">' + colorStr + '</span>';
}

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
        document.getElementById('stat-window').textContent = '-';
        document.getElementById('primary-digit').textContent = '-';
        document.getElementById('digit-pills').textContent = '---';
        document.getElementById('digit-sub').textContent = 'Analyzing digits...';
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

        var dbCount = (data.data_lineage && data.data_lineage.database_record_count) || data.database_record_count || data.total_records_analyzed || '-';
        var winCount = (data.data_lineage && data.data_lineage.feature_window_selected) || data.selected_window || data.feature_window_selected || '-';

        document.getElementById('conf-text').innerHTML = 'Confidence: <b>' + confPct + '%</b> — <span class="badge ' + badgeClass + '">' + level + '</span>';
        document.getElementById('pred-status').innerHTML = '<span class="status-dot ready"></span>Prediction Ready';
        document.getElementById('stat-signals').textContent = (data.agreeing_indicators || '-') + '/' + (data.active_indicators || '-');
        document.getElementById('stat-records').textContent = Number(dbCount).toLocaleString();
        document.getElementById('stat-window').textContent = winCount + ' draws';

        // Render Game 1: Number Game
        if (data.digit_prediction) {
          var dp = data.digit_prediction;
          if (dp.abstained) {
            document.getElementById('primary-digit').textContent = '-';
            document.getElementById('digit-pills').textContent = 'ABSTAINED';
            document.getElementById('digit-sub').textContent = 'Low edge (high entropy)';
          } else {
            var topNums = dp.top_numbers || [];
            var predDigitStr = dp.predicted_digit !== null && dp.predicted_digit !== undefined ? dp.predicted_digit : (topNums.length > 0 ? topNums[0] : '-');
            document.getElementById('primary-digit').textContent = predDigitStr;
            document.getElementById('digit-pills').textContent = topNums.join('  •  ');
            var massPct = ((dp.top4_probability_mass || 0) * 100).toFixed(1);
            document.getElementById('digit-sub').textContent = 'Top-4 Coverage: ' + massPct + '%';
          }
        }
      }
    }
    // Waiting / Stale / Gap / Error diagnostic states
    else {
      var waitKey = "wait_" + status + "_" + periodDisplay + "_" + (data.total_records_analyzed || 0);
      if (waitKey !== lastRenderedKey) {
        lastRenderedKey = waitKey;
        document.getElementById('pred-label').textContent = 'PERIOD #' + (periodDisplay !== "---" ? periodDisplay : "---");
        var pText = (status === "STALE_DATA" || status === "COLLECTOR_STALE") ? 'STALE' : 'WAIT';
        var predEl = document.getElementById('pred-text');
        predEl.textContent = pText;
        predEl.className = 'pred-val wait';
        var msg = data.message || (data.reason ? data.reason.replace(/_/g, ' ') : 'Collecting historical records...');
        document.getElementById('conf-text').textContent = msg;
        document.getElementById('pred-status').innerHTML = '<span class="status-dot waiting"></span>' + (status ? status.replace(/_/g, ' ') : 'Waiting for data');
        var dbCountWait = (data.data_lineage && data.data_lineage.database_record_count) || data.database_record_count || data.total_records_analyzed || '-';
        var winCountWait = (data.data_lineage && data.data_lineage.feature_window_selected) || data.selected_window || data.feature_window_selected || '-';
        document.getElementById('stat-signals').textContent = '-';
        document.getElementById('stat-records').textContent = Number(dbCountWait).toLocaleString();
        document.getElementById('stat-window').textContent = winCountWait + ' draws';
        document.getElementById('primary-digit').textContent = '-';
        document.getElementById('digit-pills').textContent = '---';
        document.getElementById('digit-sub').textContent = 'Waiting for data';
      }
    }

    // Real Game History section & Latest Completed Result
    if (data.recent_history && data.recent_history.length > 0) {
      var latestRec = data.recent_history[0];
      var latestOutcome = latestRec.result || latestRec.actual || latestRec.size || '---';
      var latestNumStr = (latestRec.result_number !== undefined && latestRec.result_number !== null) ? latestRec.result_number : (latestRec.number !== undefined ? latestRec.number : '-');
      var latestPeriod = String(latestRec.issue_id || latestRec.period || '');
      var latestColor = latestRec.color || latestRec.source_color || '';

      document.getElementById('latest-period-label').textContent = 'PERIOD #' + (latestPeriod || '---');
      document.getElementById('latest-num-val').textContent = latestNumStr;

      var lSizeEl = document.getElementById('latest-size-val');
      lSizeEl.textContent = latestOutcome;
      lSizeEl.className = 'tag ' + (latestOutcome === 'BIG' ? 'big' : 'small');
      document.getElementById('latest-color-val').innerHTML = formatColorHtml(latestColor);

      var hist = data.recent_history.slice(0, 10);
      var tbody = document.getElementById('history-body');
      tbody.innerHTML = hist.map(function(item) {
        var outcome = item.result || item.actual || item.size || '---';
        var actClass = outcome === 'BIG' ? 'big' : 'small';
        var numStr = (item.result_number !== undefined && item.result_number !== null) ? item.result_number : (item.number !== undefined ? item.number : '-');
        var colStr = item.color || item.source_color || '';

        return '<tr>' +
          '<td>#' + String(item.issue_id || item.period || '') + '</td>' +
          '<td><b style="color:#ffd700">' + numStr + '</b></td>' +
          '<td><span class="tag ' + actClass + '">' + outcome + '</span></td>' +
          '<td style="text-align:right">' + formatColorHtml(colStr) + '</td>' +
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

        # Attach real GameResult history (authoritative observed outcomes only — top 5)
        try:
            rows_query = await session.execute(
                select(GameResult).order_by(desc(GameResult.issue_id)).limit(5)
            )
            rows = rows_query.scalars().all()
            prediction["recent_history"] = [
                {
                    "period": r.issue_id,
                    "issue_id": r.issue_id,
                    "result": r.calculated_size,
                    "actual": r.calculated_size,
                    "result_number": r.result_number,
                    "color": r.source_color,
                }
                for r in rows
            ]
        except Exception as err:
            from app.core.logging import get_logger
            get_logger(__name__).warning("get_game_history_route_error", error=str(err))
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


@router.get("/api/v1/public/prediction/telemetry")
async def get_prediction_telemetry():
    """
    Public Read-Only Model Governance & Telemetry Endpoint.
    Exposes live rolling accuracy, Top-4 coverage, drift status, and pipeline health.
    """
    from app.analytics.telemetry import telemetry_collector
    from app.core.config import get_build_commit

    summary = telemetry_collector.get_summary_stats()
    return {
        "status": "success",
        "build_commit": get_build_commit(),
        "timestamp": int(time.time()),
        "governance": summary.get("digit_governance", {}),
        "latencies": {
            "result_to_ready": summary.get("result_to_ready_latency", {}),
            "analysis": summary.get("analysis_latency", {}),
        },
        "ai_telemetry_summary": {
            "total_requests": summary.get("ai_telemetry", {}).get("total_ai_requests", 0),
            "success_rate": summary.get("ai_telemetry", {}).get("ai_success_rate", 0.0),
            "timeout_rate": summary.get("ai_telemetry", {}).get("ai_timeout_rate", 0.0),
        },
    }

