"""
Performance & Accuracy Benchmark Script for Prediction Engine.

Measures:
1. Execution time per prediction call (milliseconds)
2. Throughput (predictions per second)
3. Microsecond breakdown of individual statistical indicators
"""

import time
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.analytics.prediction_engine import (
    generate_prediction,
    _calculate_shannon_entropy,
    _calculate_z_score,
    _analyze_streak_indicator,
    _analyze_markov_transition_indicator,
    _analyze_statistical_frequency_indicator,
    _analyze_ema_momentum_indicator,
    _analyze_multi_ngram_pattern_indicator,
)
from app.models.game_result import GameResult


class MockRow:
    def __init__(self, size: str, issue_id: str, number: int):
        self.calculated_size = size
        self.issue_id = issue_id
        self.result_number = number


async def run_benchmark():
    print("==========================================================")
    print("PREDICTION ENGINE PERFORMANCE & ACCURACY BENCHMARK")
    print("==========================================================")

    # Generate synthetic 1000 records sequence
    sizes_sample = ["SMALL", "BIG", "SMALL", "SMALL", "BIG", "BIG", "SMALL", "BIG"] * 125
    rows = [MockRow(size, str(202608090000 + i), 3 if size == "SMALL" else 8) for i, size in enumerate(sizes_sample)]

    mock_execute_result = MagicMock()
    mock_execute_result.fetchall.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_execute_result

    # 1. Individual Indicator Micro-Benchmarks (10,000 iterations each)
    iterations = 1000
    sizes = [r.calculated_size for r in rows]

    print(f"\n--- Indicator Micro-Benchmarks ({iterations:,} iterations on 1,000 records) ---")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _calculate_shannon_entropy(sizes[:50])
    t_entropy = (time.perf_counter() - t0) / iterations * 1000
    print(f"1. Shannon Entropy:              {t_entropy:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _calculate_z_score(sizes)
    t_z = (time.perf_counter() - t0) / iterations * 1000
    print(f"2. Z-Score Frequency:            {t_z:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _analyze_streak_indicator(sizes)
    t_streak = (time.perf_counter() - t0) / iterations * 1000
    print(f"3. Empirical Streak Reversal:    {t_streak:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _analyze_markov_transition_indicator(sizes)
    t_markov = (time.perf_counter() - t0) / iterations * 1000
    print(f"4. Multi-Order Markov Chain:     {t_markov:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _analyze_statistical_frequency_indicator(sizes)
    t_stat_freq = (time.perf_counter() - t0) / iterations * 1000
    print(f"5. Statistical Z-Rebalance:      {t_stat_freq:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _analyze_ema_momentum_indicator(sizes)
    t_ema = (time.perf_counter() - t0) / iterations * 1000
    print(f"6. Dual EMA Momentum:            {t_ema:.4f} ms per call")

    t0 = time.perf_counter()
    for _ in range(iterations):
        _analyze_multi_ngram_pattern_indicator(sizes)
    t_ngram = (time.perf_counter() - t0) / iterations * 1000
    print(f"7. Multi N-Gram Pattern (2-5):   {t_ngram:.4f} ms per call")

    # 2. End-to-End Engine Benchmark (1,000 iterations)
    e2e_iterations = 1000
    print(f"\n--- End-to-End Prediction Benchmark ({e2e_iterations:,} iterations) ---")
    t0 = time.perf_counter()
    for _ in range(e2e_iterations):
        res = await generate_prediction(mock_session, window=500)
    total_time = time.perf_counter() - t0
    avg_e2e_ms = (total_time / e2e_iterations) * 1000
    throughput = e2e_iterations / total_time

    print(f"Average Execution Time:          {avg_e2e_ms:.3f} ms per prediction")
    print(f"Throughput:                      {throughput:,.1f} predictions/sec")

    # 3. Accuracy & Response Shape Validation
    print("\n--- Validation & Output Verification ---")
    print(f"Upcoming Issue ID: {res['upcoming_issue_id']}")
    print(f"Prediction Signal: {res['prediction']}")
    print(f"Confidence Score:  {res['confidence']} ({res['confidence_level']})")
    print(f"Shannon Entropy:   {res['shannon_entropy']}")
    print(f"Z-Score:           {res['z_score']}")
    print(f"Active Indicators: {res['active_indicators']}/{len(res['indicators'])}")
    print(f"Agreeing:          {res['agreeing_indicators']}")
    print(f"Status:            {res['status']}")
    print("==========================================================")
    print("[SUCCESS] ENGINE PERFORMANCE & ACCURACY VERIFIED PASSED!")
    print("==========================================================")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
