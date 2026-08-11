"""
Phase 16 — Full-History Walk-Forward Research & Model Evaluation Script.

Executes deterministic walk-forward simulation across PostgreSQL GameResult data.
Prints complete research report matrix without modifying database or production code.
"""

import asyncio
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import async_session_factory
from app.analytics.walk_forward_replay import run_walk_forward_replay


class MockRow:
    def __init__(self, size, issue_id, number=5):
        self.calculated_size = size
        self.issue_id = str(issue_id)
        self.result_number = number
        self.source_color = "red" if number >= 5 else "green"


def generate_benchmark_draws(count=5000):
    rows = []
    for i in range(count):
        issue_id = str(20260800000000 + i)
        val = (i * 3 + (i // 7) * 5) % 10
        size = "BIG" if val >= 5 else "SMALL"
        rows.append(MockRow(size, issue_id, val))
    return rows


async def main():
    print("=" * 70)
    print("PHASE 16 — FULL-HISTORY WALK-FORWARD RESEARCH EXECUTION")
    print("=" * 70)

    try:
        async with async_session_factory() as session:
            report = await run_walk_forward_replay(
                session=session,
                min_history=100,
                max_eval_periods=2500,
                feature_window=1000
            )
    except Exception as err:
        print(f"[NOTE] Database connection unavailable ({err}). Running 5,000-round benchmark dataset walk-forward replay...")
        rows = generate_benchmark_draws(5000)
        report = await run_walk_forward_replay(
            rows=rows,
            min_history=100,
            max_eval_periods=2500,
            feature_window=1000
        )

    print("\n[RESULT SUMMARY]")
    print(f"Status:                      {report.get('status')}")
    print(f"Total DB Records:            {report.get('total_db_records')}")
    print(f"Evaluated Periods:           {report.get('evaluated_periods')}")
    print(f"Start Issue ID:              {report.get('start_issue_id')}")
    print(f"End Issue ID:                {report.get('end_issue_id')}")
    print(f"Replay Elapsed Latency:      {report.get('elapsed_ms')} ms")

    champ = report.get("champion_model", {})
    print("\n[CHAMPION MODEL PERFORMANCE]")
    print(f"Model Name:                  {champ.get('name')}")
    print(f"Active Evaluations:          {champ.get('active_evaluations')}")
    print(f"Correct Predictions:         {champ.get('correct_predictions')}")
    print(f"OOS Accuracy:                {champ.get('accuracy_pct')}%")
    print(f"Brier Score:                 {champ.get('brier_score')}")
    print(f"Log Loss:                    {champ.get('log_loss')}")
    print(f"Wilson 95% CI:               {champ.get('wilson_95_ci')}")
    print(f"Coverage Rate:               {champ.get('coverage_pct')}%")
    print(f"Abstention Rate:             {champ.get('abstention_pct')}%")

    base = report.get("baselines", {})
    print("\n[BENCHMARK BASELINES]")
    print(f"Random (50/50):              {base.get('random_50_50_pct')}%")
    print(f"Majority Class:              {base.get('majority_class_pct')}%")
    print(f"Last-Result Strategy:        {base.get('last_result_pct')}%")
    print(f"Markov Order-1 Strategy:     {base.get('markov_order_1_pct')}%")

    high_conf = report.get("selective_high_confidence", {})
    print("\n[SELECTIVE HIGH-CONFIDENCE PERFORMANCE]")
    print(f"Evaluations (Confluence >= 70%): {high_conf.get('evaluations')}")
    print(f"Accuracy (High Confidence):     {high_conf.get('accuracy_pct')}%")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
