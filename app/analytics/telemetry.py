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

from collections import deque


class LifecycleTelemetryCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._records: deque[dict] = deque(maxlen=max_history)
        self._ai_records: deque[dict] = deque(maxlen=max_history)
        self._provider_latencies: dict[str, list[float]] = {}

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
        # Keep latencies within max_history size
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

        # Calculate AI Telemetry Rates
        n_ai = len(self._ai_records)
        ai_successes = sum(1 for r in self._ai_records if r["success"])
        ai_timeouts = sum(1 for r in self._ai_records if r["timeout"])
        ai_errors = sum(1 for r in self._ai_records if not r["success"] and not r["timeout"])
        fallbacks = sum(1 for r in self._ai_records if r["fallback_used"])

        provider_usage: dict[str, int] = {}
        for r in self._ai_records:
            p = r["provider"]
            provider_usage[p] = provider_usage.get(p, 0) + 1

        # Extract NVIDIA and OpenRouter latencies specifically
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
