"""Public routes — Superfast lightweight prediction UI with live countdown and side-by-side prediction history."""

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
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>WinGo Predictor</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#0a0a0f;color:#ddd;font-family:system-ui,-apple-system,sans-serif;padding:10px;min-height:100vh}
    .c{max-width:420px;margin:0 auto}
    h1{font-size:14px;text-align:center;color:#888;letter-spacing:2px;text-transform:uppercase;padding:6px 0}
    .card{background:#111;border:1px solid #1e1e2e;border-radius:10px;padding:12px;margin-bottom:8px}
    .timer{text-align:center}
    .timer .lbl{font-size:10px;color:#777;text-transform:uppercase;letter-spacing:1.5px;font-weight:700}
    .timer .t{font-size:36px;font-weight:700;color:#ffd700;font-variant-numeric:tabular-nums;letter-spacing:2px;margin:2px 0}
    .timer .bar{height:3px;background:#1a1a2e;border-radius:2px;margin-top:6px}
    .timer .fill{height:100%;background:#ffd700;border-radius:2px;transition:width .3s}
    .timer .p{font-size:11px;color:#666;margin-top:6px;font-variant-numeric:tabular-nums}
    .pred{text-align:center}
    .pred .lbl{font-size:10px;color:#777;text-transform:uppercase;letter-spacing:1.5px;font-weight:700}
    .pred .sig{font-size:54px;font-weight:900;line-height:1.1;margin:2px 0}
    .pred .sig.big{color:#ff4d6a}
    .pred .sig.small{color:#4da6ff}
    .pred .sig.wait{color:#444;font-size:26px}
    .pred .conf{font-size:12px;color:#666;margin-top:2px}
    .pred .conf b{color:#ddd}
    .badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;margin-left:4px}
    .badge.h{background:#00d68f22;color:#00d68f}
    .badge.m{background:#ffd70022;color:#ffd700}
    .badge.l{background:#66666622;color:#888}
    .stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px}
    .stat{background:#111;border:1px solid #1e1e2e;border-radius:8px;padding:8px 4px;text-align:center}
    .stat .v{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
    .stat .v.g{color:#00d68f}.stat .v.y{color:#ffd700}.stat .v.b{color:#4da6ff}
    .stat .l{font-size:9px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-top:1px}
    .hdr{display:flex;justify-content:space-between;align-items:center;padding-bottom:6px;border-bottom:1px solid #1a1a2e;margin-bottom:4px}
    .hdr span{font-size:10px;color:#777;text-transform:uppercase;letter-spacing:1px;font-weight:700}
    .hdr .pct{color:#00d68f;font-weight:700;font-size:11px}
    .row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #0e0e18}
    .row:last-child{border:none}
    .row .left{display:flex;align-items:center;gap:6px}
    .num{width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:5px;font-weight:700;font-size:12px}
    .num.rb{background:#ff4d6a22;color:#ff4d6a}
    .num.sb{background:#4da6ff22;color:#4da6ff}
    .info{display:flex;flex-direction:column}
    .info .id{font-size:10px;color:#555;font-variant-numeric:tabular-nums}
    .info .sz{font-size:12px;font-weight:700}
    .info .sz.big{color:#ff4d6a}.info .sz.small{color:#4da6ff}
    .mid{display:flex;flex-direction:column;align-items:center}
    .mid .plbl{font-size:9px;color:#555;text-transform:uppercase;letter-spacing:0.5px}
    .mid .psz{font-size:11px;font-weight:700}
    .mid .psz.big{color:#ff4d6a}.mid .psz.small{color:#4da6ff}
    .w{padding:3px 8px;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap}
    .w.win{background:#00d68f18;color:#00d68f;border:1px solid #00d68f33}
    .w.loss{background:#ff4d6a18;color:#ff4d6a;border:1px solid #ff4d6a33}
    .dis{text-align:center;font-size:9px;color:#444;padding:6px}
    .urgent{animation:p .5s infinite}
    @keyframes p{50%{opacity:.4}}
  </style>
</head>
<body>
<div class="c">
  <h1>WinGo Predictor</h1>
  <div class="card timer">
    <div class="lbl">NEXT PREDICTION IN</div>
    <div class="t" id="tm">--:--</div>
    <div class="bar"><div class="fill" id="br" style="width:100%"></div></div>
    <div class="p" id="pr">Target Period #---</div>
  </div>
  <div class="card pred">
    <div class="lbl">ENGINE NEXT PREDICTION</div>
    <div class="sig wait" id="sg">---</div>
    <div class="conf" id="cf">Loading...</div>
  </div>
  <div class="stats">
    <div class="stat"><div class="v g" id="sw">-</div><div class="l">Wins/5</div></div>
    <div class="stat"><div class="v y" id="si">-</div><div class="l">Signals</div></div>
    <div class="stat"><div class="v b" id="sr">-</div><div class="l">Records</div></div>
  </div>
  <div class="card">
    <div class="hdr"><span>Real History & Prediction Match</span><span class="pct" id="ap">--%</span></div>
    <div id="hl"><div style="text-align:center;color:#444;font-size:12px;padding:16px">Loading real history...</div></div>
  </div>
  <div class="dis">100% Real Scraped Data & Empirical Statistical Ensemble Engine.</div>
</div>
<script>
function T(){
  var s=new Date().getSeconds(),r=30-(s%30),p=(r/30)*100;
  var e=document.getElementById('tm'),b=document.getElementById('br');
  e.textContent='00:'+String(r).padStart(2,'0');
  b.style.width=p+'%';
  if(r<=5){e.classList.add('urgent');b.style.background='#ff4d6a'}
  else{e.classList.remove('urgent');b.style.background='#ffd700'}
}
async function U(){
  try{
    var r=await fetch('/api/v1/public/prediction');
    var d=await r.json();
    if(d&&d.prediction){
      var s=document.getElementById('sg');
      s.textContent=d.prediction;
      s.className='sig '+d.prediction.toLowerCase();
      document.getElementById('pr').textContent='Target Period #'+(d.upcoming_issue_id||'').slice(-8);
      var cp=(d.confidence*100).toFixed(1);
      var lv=d.confidence>=.75?'HIGH':d.confidence>=.55?'MED':'LOW';
      var lc=d.confidence>=.75?'h':d.confidence>=.55?'m':'l';
      document.getElementById('cf').innerHTML='Confidence: <b>'+cp+'%</b> <span class="badge '+lc+'">'+lv+'</span>';
      document.getElementById('si').textContent=(d.active_indicators||'-')+'/'+(d.agreeing_indicators||'-');
      document.getElementById('sr').textContent=d.total_records_analyzed||'-';
    }else if(d&&d.status==='INSUFFICIENT_DATA'){
      document.getElementById('sg').textContent='WAIT';
      document.getElementById('sg').className='sig wait';
      document.getElementById('cf').innerHTML='Collecting data...';
    }
    if(d&&d.recent_history&&d.recent_history.length>0){
      var h=d.recent_history,w=h.filter(function(x){return x.is_win}).length;
      var pct=Math.round((w/h.length)*100);
      document.getElementById('sw').textContent=w+'/'+h.length;
      var ae=document.getElementById('ap');
      ae.textContent=pct+'%';
      ae.style.color=pct>=60?'#00d68f':pct>=40?'#ffd700':'#ff4d6a';
      document.getElementById('hl').innerHTML=h.map(function(i){
        var b=i.size==='BIG';
        var pb=i.predicted_size==='BIG';
        var winText=i.is_win ? '✅ WIN (' + i.predicted_size + ')' : '❌ LOSS (' + i.predicted_size + ')';
        return '<div class="row">' +
          '<div class="left">' +
            '<div class="num '+(b?'rb':'sb')+'">'+i.result+'</div>' +
            '<div class="info">' +
              '<span class="id">#'+i.issue_id.slice(-8)+'</span>' +
              '<span class="sz '+(b?'big':'small')+'">'+i.size+'</span>' +
            '</div>' +
          '</div>' +
          '<div class="mid">' +
            '<span class="plbl">PRED</span>' +
            '<span class="psz '+(pb?'big':'small')+'">'+i.predicted_size+'</span>' +
          '</div>' +
          '<span class="w '+(i.is_win?'win':'loss')+'">'+winText+'</span>' +
        '</div>';
      }).join('');
    }
  }catch(e){}
}
T();U();setInterval(T,250);setInterval(U,1000);
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_minimal_ui():
    """Superfast lightweight prediction UI with live countdown."""
    return HTML_PAGE


@router.get("/api/v1/public/prediction")
async def get_public_prediction(session: AsyncSession = Depends(get_session)):
    """Public unauthenticated prediction readout with recent accuracy verification."""
    prediction = await get_prediction(session)

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

    rows_query = await session.execute(
        select(GameResult).order_by(desc(GameResult.issue_id)).limit(20)
    )
    rows = rows_query.scalars().all()
    prediction["recent_history"] = evaluate_recent_accuracy(rows)
    return prediction
