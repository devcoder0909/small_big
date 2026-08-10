"""
V3 Adaptive Window Selector — Dynamic Historical Horizon Engine.

Evaluates historical candidate windows ([25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000])
based strictly on OUT-OF-SAMPLE (OOS) walk-forward performance, market regime alignment, calibration score,
sample sufficiency, and compute latency cost.

Zero In-Sample Leakage: Window selection is driven strictly by past confirmed outcome tracking.
"""

from collections import defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)

CANDIDATE_WINDOWS = [25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
STABLE_DEFAULT_WINDOW = 1000
MIN_EVALUATION_SAMPLES = 50


class AdaptiveWindowSelector:
    """Dynamic Adaptive Historical Window Selection Engine."""

    def __init__(self, min_samples: int = MIN_EVALUATION_SAMPLES, default_window: int = STABLE_DEFAULT_WINDOW):
        self.min_samples = min_samples
        self.default_window = default_window

        # Performance tracking structures: window -> stats
        # stats = {"wins": int, "attempts": int, "brier_sum": float}
        self._window_stats: dict[int, dict] = {
            w: {"wins": 0, "attempts": 0, "brier_sum": 0.0} for w in CANDIDATE_WINDOWS
        }
        # Regime-specific performance: (window, regime) -> stats
        self._regime_window_stats: dict[tuple[int, str], dict] = defaultdict(
            lambda: {"wins": 0, "attempts": 0, "brier_sum": 0.0}
        )
        self._total_evaluated_samples = 0

    def select_optimal_window(self, regime_name: str = "STABLE_NEUTRAL") -> tuple[int, dict]:
        """
        Select optimal historical analysis window for the current prediction cycle.

        Args:
            regime_name: Current market regime (e.g. STREAK_HEAVY, ALTERNATING).

        Returns:
            (selected_window_size, selection_metadata)
        """
        if self._total_evaluated_samples < self.min_samples:
            return self.default_window, {
                "selected_window": self.default_window,
                "reason": "insufficient_samples_using_stable_default",
                "evaluated_samples": self._total_evaluated_samples,
                "min_required": self.min_samples,
            }

        best_window = self.default_window
        best_score = -999.0
        window_scores = {}

        for w in CANDIDATE_WINDOWS:
            w_stats = self._window_stats.get(w, {"wins": 0, "attempts": 0, "brier_sum": 0.0})
            reg_stats = self._regime_window_stats.get((w, regime_name), {"wins": 0, "attempts": 0, "brier_sum": 0.0})

            # Overall stats
            tot_att = w_stats["attempts"]
            tot_wr = (w_stats["wins"] / tot_att) if tot_att > 0 else 0.50
            avg_brier = (w_stats["brier_sum"] / tot_att) if tot_att > 0 else 0.25

            # Regime stats
            reg_att = reg_stats["attempts"]
            reg_wr = (reg_stats["wins"] / reg_att) if reg_att > 0 else 0.50

            # Composite Score: 60% regime win rate + 30% overall win rate - 10% Brier penalty - latency cost
            # Latency penalty: 0.000002 per window record to prevent picking 100k without edge gain
            latency_penalty = w * 0.000002
            composite_score = (0.60 * reg_wr) + (0.30 * tot_wr) - (0.10 * avg_brier) - latency_penalty
            window_scores[w] = round(composite_score, 4)

            if composite_score > best_score:
                best_score = composite_score
                best_window = w

        return best_window, {
            "selected_window": best_window,
            "reason": "optimal_composite_score",
            "regime": regime_name,
            "evaluated_samples": self._total_evaluated_samples,
            "composite_scores": window_scores,
        }

    def record_window_result(self, window_size: int, regime_name: str, is_win: bool, brier_score: float = 0.25):
        """
        Record out-of-sample outcome for a candidate window after result confirmation.

        Args:
            window_size: Window size evaluated.
            regime_name: Regime active when prediction was generated.
            is_win: True if prediction matched actual outcome.
            brier_score: Brier calibration score (0.0 to 1.0).
        """
        if window_size not in self._window_stats:
            self._window_stats[window_size] = {"wins": 0, "attempts": 0, "brier_sum": 0.0}

        self._window_stats[window_size]["attempts"] += 1
        if is_win:
            self._window_stats[window_size]["wins"] += 1
        self._window_stats[window_size]["brier_sum"] += brier_score

        reg_key = (window_size, regime_name)
        self._regime_window_stats[reg_key]["attempts"] += 1
        if is_win:
            self._regime_window_stats[reg_key]["wins"] += 1
        self._regime_window_stats[reg_key]["brier_sum"] += brier_score

        self._total_evaluated_samples += 1

    def get_metrics(self) -> dict:
        """Get summary metrics of window selector performance."""
        return {
            "total_evaluated_samples": self._total_evaluated_samples,
            "min_required_samples": self.min_samples,
            "window_performance": {
                w: {
                    "attempts": stats["attempts"],
                    "win_rate": round(stats["wins"] / stats["attempts"], 4) if stats["attempts"] > 0 else 0.0,
                    "avg_brier": round(stats["brier_sum"] / stats["attempts"], 4) if stats["attempts"] > 0 else 0.0,
                }
                for w, stats in self._window_stats.items()
            },
        }


# Global singleton instance for pipeline lifecycle
adaptive_window_selector = AdaptiveWindowSelector()
