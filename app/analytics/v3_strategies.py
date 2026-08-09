"""
V3 Strategy Estimators — Champion / Challenger Ensemble Strategies.

Defines 7 specialized prediction strategy estimators evaluated strictly out-of-sample:
1. V2EnsembleStrategy — Standard 15-indicator adaptive ensemble.
2. MarkovFocusStrategy — Markov transition & sequence miner focus.
3. PatternFocusStrategy — Multi-NGram pattern & hash miner focus.
4. MomentumFocusStrategy — EMA momentum, Kalman filter, & digit/color momentum focus.
5. FrequencyFocusStrategy — Statistical Z-score, Bayesian posterior, & Chi-square skew focus.
6. AIAssistedStrategy — Blends statistical ensemble with LLM pattern reasoning when online.
7. ConservativeStrategy — High consensus margin threshold with strict abstention.
"""

from typing import Any


class BaseStrategy:
    """Base class for V3 prediction strategy estimators."""

    name: str = "base_strategy"
    description: str = ""

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        """
        Evaluate indicators and return (prediction, probability).

        Returns:
            ("BIG", prob), ("SMALL", prob), or (None, 0.0) for abstention.
        """
        raise NotImplementedError


class V2EnsembleStrategy(BaseStrategy):
    """Standard V2 adaptive ensemble strategy."""

    name = "v2_ensemble"
    description = "Standard 15-indicator self-learning adaptive ensemble"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        if abs(norm_small - norm_big) < 0.020:
            return None, 0.0
        if norm_small > norm_big:
            return "SMALL", norm_small
        else:
            return "BIG", norm_big


class MarkovFocusStrategy(BaseStrategy):
    """Markov transition & sequence miner focus strategy."""

    name = "markov_focus"
    description = "Weights Markov transition and sequence miner signals heavily"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        markov = indicators.get("markov_transition", {})
        seq_miner = indicators.get("sequence_hash_miner", {})

        m_pred = markov.get("prediction")
        s_pred = seq_miner.get("prediction")

        if m_pred and s_pred and m_pred == s_pred:
            prob = max(markov.get("confidence", 0.5), seq_miner.get("confidence", 0.5))
            return m_pred, prob

        return V2EnsembleStrategy().evaluate(indicators, weights, norm_small, norm_big)


class PatternFocusStrategy(BaseStrategy):
    """Multi-NGram pattern & hash miner focus strategy."""

    name = "pattern_focus"
    description = "Weights multi-length N-gram patterns and hash miner signals"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        pattern = indicators.get("pattern_match", {})
        miner = indicators.get("sequence_hash_miner", {})

        p_pred = pattern.get("prediction")
        m_pred = miner.get("prediction")

        if p_pred and m_pred and p_pred == m_pred:
            prob = max(pattern.get("confidence", 0.5), miner.get("confidence", 0.5))
            return p_pred, prob

        return V2EnsembleStrategy().evaluate(indicators, weights, norm_small, norm_big)


class MomentumFocusStrategy(BaseStrategy):
    """EMA momentum, Kalman filter, & digit/color momentum focus strategy."""

    name = "momentum_focus"
    description = "Weights momentum crossover and numeric trend signals"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        ema = indicators.get("ema_momentum", {})
        kalman = indicators.get("kalman_filter_momentum", {})

        e_pred = ema.get("prediction")
        k_pred = kalman.get("prediction")

        if e_pred and k_pred and e_pred == k_pred:
            prob = max(ema.get("confidence", 0.5), kalman.get("confidence", 0.5))
            return e_pred, prob

        return V2EnsembleStrategy().evaluate(indicators, weights, norm_small, norm_big)


class FrequencyFocusStrategy(BaseStrategy):
    """Statistical Z-score, Bayesian posterior, & Chi-square skew focus strategy."""

    name = "frequency_focus"
    description = "Weights Z-score statistical frequency rebalance and Dirichlet-Multinomial Bayesian signals"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        stat = indicators.get("stat_frequency", {})
        bayes = indicators.get("bayesian_posterior", {})

        s_pred = stat.get("prediction")
        b_pred = bayes.get("prediction")

        if s_pred and b_pred and s_pred == b_pred:
            prob = max(stat.get("confidence", 0.5), bayes.get("confidence", 0.5))
            return s_pred, prob

        return V2EnsembleStrategy().evaluate(indicators, weights, norm_small, norm_big)


class AIAssistedStrategy(BaseStrategy):
    """Blends statistical ensemble with LLM pattern reasoning when online."""

    name = "ai_assisted"
    description = "Blends ensemble decision with LLM pattern reasoning output"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        ai_ind = indicators.get("ai_pattern_reasoning", {})
        ai_pred = ai_ind.get("prediction")

        if ai_pred and ai_ind.get("confidence", 0) >= 0.70:
            if (ai_pred == "SMALL" and norm_small >= 0.48) or (ai_pred == "BIG" and norm_big >= 0.48):
                prob = norm_small if ai_pred == "SMALL" else norm_big
                return ai_pred, prob

        return V2EnsembleStrategy().evaluate(indicators, weights, norm_small, norm_big)


class ConservativeStrategy(BaseStrategy):
    """High consensus margin threshold with strict abstention."""

    name = "conservative_abstention"
    description = "Requires at least 8% score margin threshold; abstains otherwise"

    def evaluate(self, indicators: dict, weights: dict, norm_small: float, norm_big: float) -> tuple[str | None, float]:
        if abs(norm_small - norm_big) < 0.080:
            return None, 0.0
        if norm_small > norm_big:
            return "SMALL", norm_small
        else:
            return "BIG", norm_big


# Registry of all V3 strategies
STRATEGY_REGISTRY = {
    "v2_ensemble": V2EnsembleStrategy(),
    "markov_focus": MarkovFocusStrategy(),
    "pattern_focus": PatternFocusStrategy(),
    "momentum_focus": MomentumFocusStrategy(),
    "frequency_focus": FrequencyFocusStrategy(),
    "ai_assisted": AIAssistedStrategy(),
    "conservative_abstention": ConservativeStrategy(),
}
