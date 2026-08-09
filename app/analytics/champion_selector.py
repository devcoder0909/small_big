"""
V3 Champion / Challenger Selector — Dynamic Strategy Selection Engine.

Evaluates strategy performance out-of-sample across rolling historical windows (25, 50, 100, 250, 500, 1000)
and selects the Champion strategy for the current market regime using ONLY past confirmed data.
Includes Champion Drift Monitoring.
"""

from collections import deque
from app.analytics.v3_strategies import STRATEGY_REGISTRY, BaseStrategy
from app.core.logging import get_logger

logger = get_logger(__name__)

WINDOWS = [25, 50, 100, 250, 500, 1000]


class ChampionSelector:
    """Champion / Challenger strategy selection and performance tracking manager."""

    def __init__(self):
        self._strategy_wins: dict[str, int] = {k: 0 for k in STRATEGY_REGISTRY}
        self._strategy_votes: dict[str, int] = {k: 0 for k in STRATEGY_REGISTRY}
        self._regime_performance: dict[str, dict[str, list[int]]] = {}
        self._history: deque[dict] = deque(maxlen=1000)
        self._drift_status: dict = {"drift_detected": False, "message": "Champion performing optimally"}

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

        # Track rolling history
        self._history.append({
            "strategy": strategy_name,
            "regime": regime_name,
            "is_win": is_win,
        })

        self._check_champion_drift()

    def _check_champion_drift(self):
        """Check if current Champion performance has deteriorated relative to top Challenger."""
        if len(self._history) < 25:
            return

        # Check last 50 period window
        recent = list(self._history)[-50:]
        stats: dict[str, list[int]] = {}
        for item in recent:
            st = item["strategy"]
            if st not in stats:
                stats[st] = [0, 0]
            stats[st][1] += 1
            if item["is_win"]:
                stats[st][0] += 1

        champ = "v2_ensemble"
        champ_stats = stats.get(champ, [0, 0])
        champ_wr = champ_stats[0] / champ_stats[1] if champ_stats[1] > 0 else 0.50

        top_challenger = None
        top_challenger_wr = 0.0

        for st, (wins, votes) in stats.items():
            if st == champ or votes < 10:
                continue
            wr = wins / votes
            if wr > top_challenger_wr:
                top_challenger_wr = wr
                top_challenger = st

        if top_challenger and (top_challenger_wr - champ_wr) >= 0.05:
            logger.warning(
                "CHAMPION_DRIFT_DETECTED",
                champion=champ,
                champion_win_rate=round(champ_wr, 4),
                challenger=top_challenger,
                challenger_win_rate=round(top_challenger_wr, 4),
            )
            self._drift_status = {
                "drift_detected": True,
                "message": f"CHAMPION_DRIFT_DETECTED: Champion ({champ} wr={champ_wr:.2f}) underperforming Challenger ({top_challenger} wr={top_challenger_wr:.2f})",
                "champion": champ,
                "challenger": top_challenger,
                "champion_win_rate": round(champ_wr, 4),
                "challenger_win_rate": round(top_challenger_wr, 4),
            }
        else:
            self._drift_status = {"drift_detected": False, "message": "Champion performing optimally"}

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

        # Calculate rolling window win-rates
        history_list = list(self._history)
        rolling_metrics = {}
        for w in WINDOWS:
            sub = history_list[-w:] if len(history_list) >= w else history_list
            total_wins = sum(1 for x in sub if x["is_win"])
            total_votes = len(sub)
            rolling_metrics[f"last_{w}"] = {
                "total": total_votes,
                "win_rate": round(total_wins / total_votes, 4) if total_votes > 0 else 0.50,
            }

        return {
            "strategies": summary,
            "regimes": self._regime_performance,
            "rolling_windows": rolling_metrics,
            "champion_drift": self._drift_status,
        }


# Global singleton
champion_selector = ChampionSelector()
