"""Backtesting framework — evaluate statistical rules against historical data.

IMPORTANT: This is HISTORICAL BACKTEST ONLY.
Backtesting past data does NOT prove a rule will work in the future.
Past performance does NOT guarantee future results.
"""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.game_result import GameResult


class BacktestRule:
    """Base class for backtesting rules."""

    name: str = "base_rule"
    description: str = ""

    def predict(self, history: list[str]) -> str | None:
        """
        Given a history of sizes (newest first), predict the next one.

        Args:
            history: List of "BIG"/"SMALL" strings, newest first.

        Returns:
            "BIG", "SMALL", or None (no prediction).
        """
        raise NotImplementedError


class StreakReversalRule(BacktestRule):
    """Predict reversal after a streak of N same results."""

    def __init__(self, streak_threshold: int = 4):
        self.name = f"streak_reversal_{streak_threshold}"
        self.description = f"Predict reversal after {streak_threshold} consecutive same results"
        self.threshold = streak_threshold

    def predict(self, history: list[str]) -> str | None:
        if len(history) < self.threshold:
            return None
        # Check if last N results are the same
        recent = history[:self.threshold]
        if all(s == recent[0] for s in recent):
            return "BIG" if recent[0] == "SMALL" else "SMALL"
        return None


class TransitionRule(BacktestRule):
    """Predict based on most common transition from current state."""

    def __init__(self):
        self.name = "transition_follow"
        self.description = "Follow most common historical transition"

    def predict(self, history: list[str]) -> str | None:
        if len(history) < 20:
            return None
        current = history[0]

        # Count transitions from current state in history
        same = 0
        opposite = 0
        for i in range(1, len(history) - 1):
            if history[i + 1] == current:  # i+1 is "from", i is "to"
                if history[i] == current:
                    same += 1
                else:
                    opposite += 1

        if same + opposite == 0:
            return None
        return current if same > opposite else ("BIG" if current == "SMALL" else "SMALL")


class FrequencyRebalanceRule(BacktestRule):
    """Predict the underrepresented side when deviation exceeds threshold."""

    def __init__(self, window: int = 50, threshold_pct: float = 10.0):
        self.name = f"frequency_rebalance_{window}_{threshold_pct}"
        self.description = f"Predict underrepresented side when deviation > {threshold_pct}% in last {window}"
        self.window = window
        self.threshold = threshold_pct

    def predict(self, history: list[str]) -> str | None:
        if len(history) < self.window:
            return None
        window_data = history[:self.window]
        small_count = sum(1 for s in window_data if s == "SMALL")
        small_pct = small_count / self.window * 100

        if small_pct > 50 + self.threshold:
            return "BIG"
        elif small_pct < 50 - self.threshold:
            return "SMALL"
        return None


async def run_backtest(
    session: AsyncSession,
    rules: list[BacktestRule] | None = None,
    sample_size: int = 1000,
) -> list[dict]:
    """
    Run backtesting on historical data.

    HISTORICAL BACKTEST ONLY — results do NOT predict future outcomes.

    Args:
        session: Database session.
        rules: List of BacktestRule instances. Uses defaults if None.
        sample_size: Number of historical records to test.

    Returns:
        List of backtest results per rule.
    """
    if rules is None:
        rules = [
            StreakReversalRule(3),
            StreakReversalRule(4),
            StreakReversalRule(5),
            TransitionRule(),
            FrequencyRebalanceRule(50, 8),
            FrequencyRebalanceRule(100, 10),
        ]

    # Fetch historical data (oldest first for chronological backtesting)
    query = (
        select(GameResult.calculated_size)
        .order_by(desc(GameResult.issue_id))
        .limit(sample_size)
    )
    result = await session.execute(query)
    rows = result.fetchall()

    if len(rows) < 50:
        return [{
            "error": "Insufficient data for backtesting",
            "records_available": len(rows),
            "minimum_required": 50,
        }]

    # Reverse to get chronological order (oldest first)
    all_sizes = [row.calculated_size for row in reversed(rows)]

    results = []

    for rule in rules:
        correct = 0
        incorrect = 0
        no_prediction = 0

        # Walk through history, using past data to predict next
        for i in range(50, len(all_sizes)):
            # Build history as newest-first (reversed slice up to current position)
            history = list(reversed(all_sizes[:i]))
            prediction = rule.predict(history)

            if prediction is None:
                no_prediction += 1
            elif prediction == all_sizes[i]:
                correct += 1
            else:
                incorrect += 1

        total_predictions = correct + incorrect
        accuracy = round(correct / total_predictions * 100, 2) if total_predictions > 0 else 0

        results.append({
            "rule_name": rule.name,
            "description": rule.description,
            "total_samples": len(all_sizes) - 50,
            "predictions_made": total_predictions,
            "no_prediction": no_prediction,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_pct": accuracy,
            "coverage_pct": round(total_predictions / (len(all_sizes) - 50) * 100, 2)
            if len(all_sizes) > 50 else 0,
            "disclaimer": "HISTORICAL BACKTEST ONLY — past performance does NOT guarantee future results",
        })

    return results
