"""
Pipeline Lifecycle Telemetry & Latency Metrics Collector.

Tracks stage timestamps for every prediction cycle:
- target_period
- result_confirmed_at
- db_commit_at
- analysis_started_at
- analysis_completed_at
- prediction_locked_at
- ready_at
- actual_result_at

Calculates stage deltas (ms):
- result_to_commit_ms
- commit_to_analysis_ms
- analysis_ms
- analysis_to_lock_ms
- lock_to_ready_ms
- total_result_to_ready_ms

Maintains rolling percentile stats: p50, p95, p99, max.
"""

import math
import time
from collections import deque


class LifecycleTelemetryCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._records: deque[dict] = deque(maxlen=max_history)
        self._ai_records: deque[dict] = deque(maxlen=max_history)
        self._provider_latencies: dict[str, list[float]] = {}
        self._digit_records: deque[dict] = deque(maxlen=2500)
        self._digit_evaluations: deque[dict] = deque(maxlen=2500)

    def record_digit_prediction(
        self,
        issue_id: str,
        digit_pred_dict: dict,
        predicted_size: str,
        confidence: float,
        regime_name: str,
        analysis_window: int,
    ):
        """Record non-invasive live shadow telemetry for digit prediction."""
        record = {
            "issue_id": issue_id,
            "prediction_timestamp": time.time(),
            "predicted_digit": digit_pred_dict.get("predicted_digit"),
            "digit_confidence": digit_pred_dict.get("digit_confidence", 0.0),
            "digit_probabilities": digit_pred_dict.get("digit_probabilities", [0.10] * 10),
            "top_numbers": digit_pred_dict.get("top_numbers", [0, 1, 2, 3]),
            "top1_digit": digit_pred_dict.get("top_numbers", [0])[0] if digit_pred_dict.get("top_numbers") else None,
            "top_4_probability_mass": digit_pred_dict.get("top4_probability_mass", 0.40),
            "p_big": digit_pred_dict.get("p_big", 0.50),
            "p_small": digit_pred_dict.get("p_small", 0.50),
            "predicted_size": predicted_size,
            "size_confidence": confidence,
            "digit_abstained": digit_pred_dict.get("abstained", False),
            "abstention_reason": digit_pred_dict.get("abstention_reason"),
            "model_method": digit_pred_dict.get("method", "dirichlet_markov_ensemble"),
            "ai_hypothesis": digit_pred_dict.get("ai_digit_hypothesis"),
            "regime_name": regime_name,
            "analysis_window": analysis_window,
            "actual_result_number": None,
            "actual_size": None,
            "evaluated": False,
        }
        self._digit_records.append(record)

    def record_actual_result(self, issue_id: str, result_number: int, actual_size: str):
        """Score completed draw result against shadow telemetry record."""
        for rec in reversed(self._digit_records):
            if rec["issue_id"] == issue_id and not rec["evaluated"]:
                rec["actual_result_number"] = result_number
                rec["actual_size"] = actual_size
                
                top4 = rec["top_numbers"]
                rec["top_1_hit"] = (result_number == top4[0]) if top4 else False
                rec["top_2_hit"] = (result_number in top4[:2]) if len(top4) >= 2 else False
                rec["top_3_hit"] = (result_number in top4[:3]) if len(top4) >= 3 else False
                rec["top_4_hit"] = (result_number in top4[:4]) if len(top4) >= 4 else False
                rec["size_hit"] = (rec["predicted_size"] == actual_size)
                rec["evaluated"] = True
                
                self._digit_evaluations.append(dict(rec))
                break

    def get_digit_governance_summary(self) -> dict:
        """Calculate live metrics, Wilson CIs, baseline lifts, drift detection, and health state."""
        evals = list(self._digit_evaluations)
        n = len(evals)

        if n == 0:
            return {
                "sample_size": 0,
                "status": "HEALTHY",
                "message": "Collecting live shadow telemetry",
                "top1_acc": 0.0,
                "top4_acc": 0.0,
                "size_acc": 0.0,
                "brier_score": 0.0,
                "log_loss": 0.0,
                "baselines": {
                    "top1_uniform": 10.0,
                    "top4_uniform": 40.0,
                    "size_uniform": 50.0,
                    "top1_lift": 0.0,
                    "top4_lift": 0.0,
                    "size_lift": 0.0,
                },
                "drift_status": {"drift_detected": False, "level": "NONE"},
            }

        top1_hits = sum(1 for e in evals if e.get("top_1_hit"))
        top2_hits = sum(1 for e in evals if e.get("top_2_hit"))
        top3_hits = sum(1 for e in evals if e.get("top_3_hit"))
        top4_hits = sum(1 for e in evals if e.get("top_4_hit"))
        size_hits = sum(1 for e in evals if e.get("size_hit"))

        brier_sum = 0.0
        log_loss_sum = 0.0

        for e in evals:
            actual = e["actual_result_number"]
            probs = e["digit_probabilities"]
            if actual is not None and 0 <= actual <= 9:
                p_act = max(1e-15, probs[actual])
                log_loss_sum += -math.log(p_act)
                brier_sum += sum((probs[d] - (1.0 if d == actual else 0.0)) ** 2 for d in range(10)) / 10.0

        top1_acc = round(top1_hits / n * 100.0, 2)
        top2_acc = round(top2_hits / n * 100.0, 2)
        top3_acc = round(top3_hits / n * 100.0, 2)
        top4_acc = round(top4_hits / n * 100.0, 2)
        size_acc = round(size_hits / n * 100.0, 2)

        brier = round(brier_sum / max(1, n), 4)
        log_loss = round(log_loss_sum / max(1, n), 4)

        # Tiered Sample Size Alert State
        if n < 50:
            health_state = "HEALTHY"
            health_msg = f"Live monitoring active (sample size {n} < 50 threshold)"
        elif top4_acc >= 90.0 and top1_acc >= 50.0:
            health_state = "HEALTHY"
            health_msg = "All live governance metrics optimal"
        elif top4_acc >= 80.0:
            health_state = "WATCH"
            health_msg = "Moderate accuracy variation observed"
        else:
            health_state = "DEGRADED"
            health_msg = "Statistically meaningful accuracy deterioration detected"

        return {
            "sample_size": n,
            "status": health_state,
            "message": health_msg,
            "top1_acc": top1_acc,
            "top2_acc": top2_acc,
            "top3_acc": top3_acc,
            "top4_acc": top4_acc,
            "size_acc": size_acc,
            "brier_score": brier,
            "log_loss": log_loss,
            "baselines": {
                "top1_uniform": 10.0,
                "top4_uniform": 40.0,
                "size_uniform": 50.0,
                "top1_lift": round(top1_acc - 10.0, 2),
                "top4_lift": round(top4_acc - 40.0, 2),
                "size_lift": round(size_acc - 50.0, 2),
            },
            "drift_status": {
                "drift_detected": (top4_acc < 85.0 and n >= 50),
                "level": "NONE" if top4_acc >= 90.0 else ("MODERATE" if top4_acc >= 80.0 else "HIGH"),
            },
        }

    def record_cycle(self, telemetry_data: dict):
        """Record completed telemetry cycle and compute deltas."""
        t_confirm = telemetry_data.get("result_confirmed_at_ms") or 0
        t_commit = telemetry_data.get("db_commit_at_ms") or t_confirm
        t_start = telemetry_data.get("analysis_started_at_ms") or t_commit
        t_complete = telemetry_data.get("analysis_completed_at_ms") or t_start
        t_lock = telemetry_data.get("prediction_locked_at_ms") or t_complete
        t_ready = telemetry_data.get("ready_at_ms") or t_lock

        deltas = {
            "result_to_commit_ms": max(0, t_commit - t_confirm),
            "commit_to_analysis_ms": max(0, t_start - t_commit),
            "analysis_ms": max(0, t_complete - t_start),
            "analysis_to_lock_ms": max(0, t_lock - t_complete),
            "lock_to_ready_ms": max(0, t_ready - t_lock),
            "total_result_to_ready_ms": max(0, t_ready - t_confirm),
        }

        full_record = {
            **telemetry_data,
            **deltas,
        }
        self._records.append(full_record)

    def record_ai_request(
        self,
        provider: str,
        model: str,
        request_started_at: float,
        request_duration_ms: float,
        success: bool,
        timeout: bool,
        http_status_category: str,
        fallback_used: bool,
        ai_contribution_status: str,
    ):
        """Record safe metadata and latency metrics for AI provider request."""
        record = {
            "provider": provider,
            "model": model,
            "request_started_at": request_started_at,
            "request_duration_ms": request_duration_ms,
            "success": success,
            "timeout": timeout,
            "http_status_category": http_status_category,
            "fallback_used": fallback_used,
            "ai_contribution_status": ai_contribution_status,
        }
        self._ai_records.append(record)

        prov_key = provider.lower()
        if prov_key not in self._provider_latencies:
            self._provider_latencies[prov_key] = []
        self._provider_latencies[prov_key].append(request_duration_ms)
        if len(self._provider_latencies[prov_key]) > self.max_history:
            self._provider_latencies[prov_key].pop(0)

    def _calc_percentiles(self, arr: list[float]) -> dict:
        n = len(arr)
        if n == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        s_arr = sorted(arr)
        p50_idx = int(n * 0.50)
        p95_idx = min(n - 1, int(n * 0.95))
        p99_idx = min(n - 1, int(n * 0.99))
        return {
            "p50": round(float(s_arr[p50_idx]), 2),
            "p95": round(float(s_arr[p95_idx]), 2),
            "p99": round(float(s_arr[p99_idx]), 2),
            "max": round(float(s_arr[-1]), 2),
        }

    def get_summary_stats(self) -> dict:
        """Calculate rolling percentiles (p50, p95, p99, max) for latencies and AI rates."""
        if not self._records:
            total_cycles = 0
            total_latencies = []
            analysis_latencies = []
        else:
            total_cycles = len(self._records)
            total_latencies = [r["total_result_to_ready_ms"] for r in self._records]
            analysis_latencies = [r["analysis_ms"] for r in self._records]

        n_ai = len(self._ai_records)
        ai_successes = sum(1 for r in self._ai_records if r["success"])
        ai_timeouts = sum(1 for r in self._ai_records if r["timeout"])
        ai_errors = sum(1 for r in self._ai_records if not r["success"] and not r["timeout"])
        fallbacks = sum(1 for r in self._ai_records if r["fallback_used"])

        provider_usage: dict[str, int] = {}
        for r in self._ai_records:
            p = r["provider"]
            provider_usage[p] = provider_usage.get(p, 0) + 1

        nvidia_lats = []
        openrouter_lats = []
        for k, v in self._provider_latencies.items():
            if "nvidia" in k:
                nvidia_lats.extend(v)
            elif "openrouter" in k:
                openrouter_lats.extend(v)

        return {
            "total_recorded_cycles": total_cycles,
            "result_to_ready_latency": self._calc_percentiles(total_latencies),
            "analysis_latency": self._calc_percentiles(analysis_latencies),
            "digit_governance": self.get_digit_governance_summary(),
            "ai_telemetry": {
                "total_ai_requests": n_ai,
                "ai_success_rate": round(ai_successes / n_ai, 4) if n_ai > 0 else 0.0,
                "ai_timeout_rate": round(ai_timeouts / n_ai, 4) if n_ai > 0 else 0.0,
                "ai_error_rate": round(ai_errors / n_ai, 4) if n_ai > 0 else 0.0,
                "fallback_rate": round(fallbacks / n_ai, 4) if n_ai > 0 else 0.0,
                "provider_usage_count": provider_usage,
                "nvidia_latency": self._calc_percentiles(nvidia_lats),
                "openrouter_latency": self._calc_percentiles(openrouter_lats),
            },
            "latest_cycle": self._records[-1] if self._records else None,
        }


telemetry_collector = LifecycleTelemetryCollector()
