"""
Deterministic Historical Walk-Forward Replay Evaluator (Phase 16-C).

Simulates exact chronological prediction performance across the entire historical GameResult dataset.
Enforces strictly ZERO future leakage: for target period T+1, only GameResult rows <= T are visible.
"""

import math
import time
from typing import Dict, List, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game_result import GameResult
from app.analytics.prediction_engine import (
    _run_all_indicators,
    _calculate_adaptive_indicator_weights,
    _score_indicators,
    _calculate_shannon_entropy,
    DEFAULT_WEIGHTS,
)


class MockRow:
    """Mock row wrapper for GameResult data matching engine interface."""
    def __init__(self, calculated_size: str, issue_id: str, result_number: int, source_color: str = "red"):
        self.calculated_size = calculated_size
        self.issue_id = issue_id
        self.result_number = result_number
        self.source_color = source_color


def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """Calculate 95% Wilson Score Interval for binomial proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    lower = max(0.0, round((centre - spread) * 100, 2))
    upper = min(100.0, round((centre + spread) * 100, 2))
    return [lower, upper]


async def run_walk_forward_replay(
    session: AsyncSession | None = None,
    rows: List[Any] | None = None,
    min_history: int = 100,
    max_eval_periods: int = 2500,
    feature_window: int = 1000
) -> Dict[str, Any]:
    """
    Perform a complete chronological walk-forward replay on GameResult history.
    """
    t0 = time.monotonic()

    if rows is not None:
        all_db_rows = rows
    elif session is not None:
        query = select(GameResult).order_by(GameResult.issue_id.asc())
        result = await session.execute(query)
        all_db_rows = result.scalars().all()
    else:
        all_db_rows = []

    total_db_records = len(all_db_rows)
    if total_db_records < min_history + 10:
        return {
            "status": "INSUFFICIENT_HISTORICAL_DATA",
            "total_db_records": total_db_records,
            "min_history_required": min_history,
        }

    # Prepare mock rows list
    all_rows = [
        MockRow(
            calculated_size=row.calculated_size,
            issue_id=row.issue_id,
            result_number=row.result_number,
            source_color=getattr(row, "source_color", "red")
        )
        for row in all_db_rows
    ]

    # Replay evaluation bounds
    start_idx = max(min_history, total_db_records - max_eval_periods)
    end_idx = total_db_records

    evaluations = []
    
    # Baseline tracking
    baseline_random_correct = 0
    baseline_majority_correct = 0
    baseline_last_result_correct = 0
    
    # Champion Model tracking
    champ_correct = 0
    champ_active_count = 0
    champ_brier_scores = []
    champ_log_losses = []
    
    # Model B (Markov Order-1) tracking
    markov_correct = 0
    markov_total = 0
    
    # Selective prediction thresholds
    high_conf_correct = 0
    high_conf_total = 0

    for i in range(start_idx, end_idx):
        target_row = all_rows[i]
        actual_outcome = target_row.calculated_size
        target_issue_id = target_row.issue_id

        # STRICT ISOLATION: History visible up to i-1 (recent rows DESC order for feature builder)
        history_slice = list(reversed(all_rows[max(0, i - feature_window):i]))

        if len(history_slice) < 5:
            continue

        sizes = [r.calculated_size for r in history_slice]
        numbers = [r.result_number for r in history_slice]
        colors = [r.source_color for r in history_slice]

        # Extract all 15 indicators
        indicators = _run_all_indicators(sizes, numbers, colors)
        weights = _calculate_adaptive_indicator_weights(sizes, DEFAULT_WEIGHTS, numbers, colors)
        small_w, big_w, total_w, active_indicators = _score_indicators(indicators, weights)

        tot_score = small_w + big_w
        if tot_score > 0:
            norm_small = small_w / tot_score
            norm_big = big_w / tot_score
        else:
            norm_small, norm_big = 0.5, 0.5

        pred_size = "BIG" if norm_big > norm_small else "SMALL"
        shannon_entropy = _calculate_shannon_entropy(sizes[:50])
        confidence = max(norm_big, norm_small)

        # Signal determination
        agreeing = sum(1 for ind in indicators.values() if ind.get("prediction") == pred_size)
        agreement_pct = (agreeing / active_indicators * 100.0) if active_indicators > 0 else 50.0
        is_active_signal = agreement_pct >= 60.0 and confidence >= 0.55

        # Brier & Log Loss
        actual_val = 1.0 if actual_outcome == "BIG" else 0.0
        prob_big = norm_big
        brier = (actual_val - prob_big) ** 2
        log_loss = -math.log(max(1e-5, prob_big if actual_outcome == "BIG" else 1.0 - prob_big))

        # Champion evaluation
        is_correct = (pred_size == actual_outcome)
        if is_active_signal:
            champ_active_count += 1
            if is_correct:
                champ_correct += 1
            champ_brier_scores.append(brier)
            champ_log_losses.append(log_loss)

        # Baseline evaluation
        # Baseline 1: Majority Class (BIG)
        if actual_outcome == "BIG":
            baseline_majority_correct += 1

        # Baseline 2: Last Result
        last_res = sizes[0] if sizes else "BIG"
        if last_res == actual_outcome:
            baseline_last_result_correct += 1

        # Baseline 3: Markov Order-1
        if len(sizes) >= 2:
            prev = sizes[0]
            # Simple Markov 1 transition count
            trans_big = sum(1 for j in range(1, len(sizes)) if sizes[j] == prev and sizes[j-1] == "BIG")
            trans_small = sum(1 for j in range(1, len(sizes)) if sizes[j] == prev and sizes[j-1] == "SMALL")
            markov_pred = "BIG" if trans_big >= trans_small else "SMALL"
            markov_total += 1
            if markov_pred == actual_outcome:
                markov_correct += 1

        # High Confidence Selective Prediction (Confluence >= 70%)
        if agreement_pct >= 70.0 and confidence >= 0.60:
            high_conf_total += 1
            if is_correct:
                high_conf_correct += 1

        evaluations.append({
            "target_issue_id": target_issue_id,
            "prediction": pred_size,
            "actual": actual_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "brier": brier,
            "log_loss": log_loss,
            "is_active_signal": is_active_signal,
        })

    eval_total = len(evaluations)
    active_eval_count = champ_active_count if champ_active_count > 0 else 1

    champ_acc = (champ_correct / active_eval_count * 100.0) if champ_active_count > 0 else 50.0
    champ_mean_brier = float(sum(champ_brier_scores) / len(champ_brier_scores)) if champ_brier_scores else 0.25
    champ_mean_log_loss = float(sum(champ_log_losses) / len(champ_log_losses)) if champ_log_losses else 0.6931
    wilson_ci = calculate_wilson_ci(champ_correct, champ_active_count)

    majority_acc = (baseline_majority_correct / eval_total * 100.0) if eval_total > 0 else 50.0
    last_result_acc = (baseline_last_result_correct / eval_total * 100.0) if eval_total > 0 else 50.0
    markov_acc = (markov_correct / markov_total * 100.0) if markov_total > 0 else 50.0
    high_conf_acc = (high_conf_correct / high_conf_total * 100.0) if high_conf_total > 0 else 0.0

    coverage_rate = (champ_active_count / eval_total * 100.0) if eval_total > 0 else 0.0
    abstention_rate = 100.0 - coverage_rate
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    return {
        "status": "COMPLETED",
        "total_db_records": total_db_records,
        "evaluated_periods": eval_total,
        "start_issue_id": all_rows[start_idx].issue_id if start_idx < total_db_records else None,
        "end_issue_id": all_rows[-1].issue_id if all_rows else None,
        "champion_model": {
            "name": "15-Indicator Bayesian Ensemble",
            "active_evaluations": champ_active_count,
            "correct_predictions": champ_correct,
            "accuracy_pct": round(champ_acc, 2),
            "brier_score": round(champ_mean_brier, 4),
            "log_loss": round(champ_mean_log_loss, 4),
            "wilson_95_ci": wilson_ci,
            "coverage_pct": round(coverage_rate, 2),
            "abstention_pct": round(abstention_rate, 2),
        },
        "baselines": {
            "random_50_50_pct": 50.0,
            "majority_class_pct": round(majority_acc, 2),
            "last_result_pct": round(last_result_acc, 2),
            "markov_order_1_pct": round(markov_acc, 2),
        },
        "selective_high_confidence": {
            "evaluations": high_conf_total,
            "correct": high_conf_correct,
            "accuracy_pct": round(high_conf_acc, 2) if high_conf_total > 0 else None,
        },
        "elapsed_ms": round(elapsed_ms, 2),
    }
