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

    def get_summary_stats(self) -> dict:
        """Calculate rolling percentiles (p50, p95, p99, max) for latencies."""
        if not self._records:
            return {
                "total_recorded_cycles": 0,
                "result_to_ready_latency": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
                "analysis_latency": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
            }

        total_latencies = sorted([r["total_result_to_ready_ms"] for r in self._records])
        analysis_latencies = sorted([r["analysis_ms"] for r in self._records])

        def calc_percentiles(arr: list[float]) -> dict:
            n = len(arr)
            if n == 0:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
            p50_idx = int(n * 0.50)
            p95_idx = min(n - 1, int(n * 0.95))
            p99_idx = min(n - 1, int(n * 0.99))
            return {
                "p50": round(float(arr[p50_idx]), 2),
                "p95": round(float(arr[p95_idx]), 2),
                "p99": round(float(arr[p99_idx]), 2),
                "max": round(float(arr[-1]), 2),
            }

        return {
            "total_recorded_cycles": len(self._records),
            "result_to_ready_latency": calc_percentiles(total_latencies),
            "analysis_latency": calc_percentiles(analysis_latencies),
            "latest_cycle": self._records[-1] if self._records else None,
        }


telemetry_collector = LifecycleTelemetryCollector()
