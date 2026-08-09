"""
V3 Champion / Challenger Selector — Dynamic Strategy Selection Engine.

Evaluates strategy performance out-of-sample across rolling historical windows (25, 50, 100, 250, 500, 1000)
and selects the Champion strategy for the current market regime using ONLY past confirmed data.
"""

from typing import Any
from app.analytics.v3_strategies import STRATEGY_REGISTRY, BaseStrategy, V2EnsembleStrategy


class ChampionSelector:
    """Champion / Challenger strategy selection and performance tracking manager."""

    def __init__(self):
        self._strategy_wins: dict[str, int] = {k: 0 for k in STRATEGY_REGISTRY}
        self._strategy_votes: dict[str, int] = {k: 0 for k in STRATEGY_REGISTRY}
        self._regime_performance: dict[str, dict[str, list[int]]] = {}

    def select_champion_strategy(
        self, regime_name: str, indicators: dict, weights: dict, norm_small: float, norm_big: float
    ) -> tuple[BaseStrategy, str | None, float]:
        """
        Select the best performing strategy for the current regime based on historical out-of-sample win-rate.

        Args:
            regime_name: Market regime name (e.g. STREAK_HEAVY, ALTERNATING).
            indicators: 15 statistical indicators dict.
            weights: Dynamic indicator weights.
            norm_small: Normalized small score.
            norm_big: Normalized big score.

        Returns:
            (selected_strategy, prediction, probability)
        """
        # Default to V2EnsembleStrategy as current benchmark
        best_strategy = STRATEGY_REGISTRY["v2_ensemble"]
        best_win_rate = 0.50

        regime_stats = self._regime_performance.get(regime_name, {})
        for name, strat in STRATEGY_REGISTRY.items():
            wins_votes = regime_stats.get(name, [0, 0])
            wins, votes = wins_votes[0], wins_votes[1]
            if votes >= 10:
                wr = wins / votes
                if wr > best_win_rate:
                    best_win_rate = wr
                    best_strategy = strat

        pred, prob = best_strategy.evaluate(indicators, weights, norm_small, norm_big)
        return best_strategy, pred, prob

    def record_result(self, strategy_name: str, regime_name: str, is_win: bool):
        """Record out-of-sample result for a completed period."""
        if strategy_name not in self._strategy_votes:
            self._strategy_votes[strategy_name] = 0
            self._strategy_wins[strategy_name] = 0

        self._strategy_votes[strategy_name] += 1
        if is_win:
            self._strategy_wins[strategy_name] += 1

        if regime_name not in self._regime_performance:
            self._regime_performance[regime_name] = {}

        if strategy_name not in self._regime_performance[regime_name]:
            self._regime_performance[regime_name][strategy_name] = [0, 0]

        self._regime_performance[regime_name][strategy_name][1] += 1
        if is_win:
            self._regime_performance[regime_name][strategy_name][0] += 1

    def get_metrics_summary(self) -> dict:
        """Return metrics summary for analytics dashboard."""
        summary = {}
        for name in STRATEGY_REGISTRY:
            votes = self._strategy_votes.get(name, 0)
            wins = self._strategy_wins.get(name, 0)
            summary[name] = {
                "votes": votes,
                "wins": wins,
                "win_rate": round(wins / votes, 4) if votes > 0 else 0.50,
            }
        return {
            "strategies": summary,
            "regimes": self._regime_performance,
        }


# Global singleton
champion_selector = ChampionSelector()
