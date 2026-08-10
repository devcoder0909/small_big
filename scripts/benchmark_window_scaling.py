"""
Historical Window Scaling & Out-Of-Sample Walk-Forward Benchmark.

Benchmarks candidate historical analysis windows:
- 500
- 1,000
- 2,000
- 5,000
- 10,000
- 25,000
- 50,000

Measures:
1. Prediction Latency (Avg, p50, p95, p99 in ms)
2. RAM Memory Footprint (Peak RAM delta in MB)
3. Out-Of-Sample Walk-Forward Predictive Accuracy (%)
4. Brier Score (Probability Calibration Loss)
5. Expected Calibration Error (ECE)
6. Abstention Rate (%)
7. Regime & Champion Stability
"""

import sys
import time
import math
import random
import asyncio
import tracemalloc
from unittest.mock import AsyncMock, MagicMock

from app.analytics.prediction_engine import generate_prediction, _run_all_indicators
from app.analytics.regime_detector import detect_market_regime
from app.analytics.champion_selector import ChampionSelector
import app.analytics.ai_rotator as ai_rotator_mod


# Mock AI rotator to prevent network LLM delays during scaling benchmark
async def _mock_fetch_ai(*args, **kwargs):
    return None

ai_rotator_mod.fetch_ai_prediction = _mock_fetch_ai


def generate_synthetic_history(count: int, seed: int = 42) -> tuple[list[str], list[int], list[str]]:
    """Generate realistic synthetic draw sequence with streaks and regime shifts."""
    rng = random.Random(seed)
    sizes = []
    numbers = []
    colors = []

    curr_size = "SMALL" if rng.random() < 0.5 else "BIG"
    streak = 0

    for _ in range(count):
        # 80% chance to stay in pattern, 20% to switch
        if streak >= 4:
            prob_continue = 0.40  # streak break likelihood
        else:
            prob_continue = 0.52

        if rng.random() > prob_continue:
            curr_size = "BIG" if curr_size == "SMALL" else "SMALL"
            streak = 1
        else:
            streak += 1

        sizes.append(curr_size)

        if curr_size == "SMALL":
            num = rng.choice([0, 1, 2, 3, 4])
        else:
            num = rng.choice([5, 6, 7, 8, 9])
        numbers.append(num)

        col = "green" if num in (1, 3, 7, 9) else ("red" if num in (2, 4, 6, 8) else "violet")
        colors.append(col)

    return sizes, numbers, colors


def mock_session_for_history(sizes: list[str], numbers: list[int], colors: list[str]):
    """Construct mock AsyncSession returning specified history."""
    rows = []
    for i in range(len(sizes)):
        issue_id = str(1000000 + len(sizes) - i)
        row = MagicMock()
        row.calculated_size = sizes[i]
        row.issue_id = issue_id
        row.result_number = numbers[i]
        row.source_color = colors[i]
        rows.append(row)

    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    session = AsyncMock()
    session.execute.return_value = mock_result
    return session


async def benchmark_window(window_size: int | None, walk_forward_evals: int = 60) -> dict:
    """Run full benchmark suite for a candidate window size."""
    effective_window = 1000 if window_size is None else window_size
    # Generate full dataset: effective_window + walk_forward_evals
    full_sizes, full_numbers, full_colors = generate_synthetic_history(effective_window + walk_forward_evals + 50, seed=123)

    # 1. Measure Latency & RAM Footprint
    sample_session = mock_session_for_history(
        full_sizes[:effective_window],
        full_numbers[:effective_window],
        full_colors[:effective_window]
    )

    tracemalloc.start()
    latencies = []

    # Warmup run
    await generate_prediction(sample_session, window_size)

    for _ in range(30):
        t0 = time.monotonic()
        pred_res = await generate_prediction(sample_session, window_size)
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies.sort()
    n_lat = len(latencies)
    avg_lat = sum(latencies) / n_lat
    p50_lat = latencies[int(n_lat * 0.50)]
    p95_lat = latencies[min(n_lat - 1, int(n_lat * 0.95))]
    p99_lat = latencies[min(n_lat - 1, int(n_lat * 0.99))]
    peak_ram_mb = peak_mem / (1024 * 1024)

    # 2. Strict Walk-Forward Evaluation (Target period = latest + 1)
    correct_predictions = 0
    total_decided = 0
    abstentions = 0
    brier_sum = 0.0
    calib_bins = {i: [0, 0] for i in range(10)}  # bin 0.5-0.6, 0.6-0.7...

    champ_sel = ChampionSelector()
    regime_switches = 0
    last_regime = None

    for idx in range(walk_forward_evals):
        # Window strictly up to draw idx (newest first)
        eval_sizes = full_sizes[idx : idx + effective_window]
        eval_numbers = full_numbers[idx : idx + effective_window]
        eval_colors = full_colors[idx : idx + effective_window]

        target_actual = full_sizes[max(0, idx - 1)] if idx > 0 else full_sizes[0]

        session = mock_session_for_history(eval_sizes, eval_numbers, eval_colors)
        pred_res = await generate_prediction(session, window_size)

        predicted_side = pred_res.get("prediction")
        conf = pred_res.get("confidence", 0.50)

        regime_data = detect_market_regime(eval_sizes)
        regime_name = regime_data["regime"]
        if last_regime and regime_name != last_regime:
            regime_switches += 1
        last_regime = regime_name

        if not predicted_side:
            abstentions += 1
        else:
            total_decided += 1
            is_win = (predicted_side == target_actual)
            if is_win:
                correct_predictions += 1

            champ_sel.record_result("v2_ensemble", regime_name, is_win)

            # Brier Score calculation
            p_target = conf if predicted_side == target_actual else (1.0 - conf)
            brier_sum += (1.0 - p_target) ** 2

            # Calibration binning
            bin_idx = min(9, int((conf - 0.50) * 20))
            calib_bins[bin_idx][1] += 1
            if is_win:
                calib_bins[bin_idx][0] += 1

    accuracy = (correct_predictions / total_decided * 100.0) if total_decided > 0 else 0.0
    brier_score = (brier_sum / total_decided) if total_decided > 0 else 0.50
    abstention_rate = (abstentions / walk_forward_evals * 100.0)

    # Expected Calibration Error
    ece = 0.0
    for b_idx, (b_wins, b_total) in calib_bins.items():
        if b_total > 0:
            bin_conf = 0.50 + (b_idx + 0.5) * 0.05
            bin_acc = b_wins / b_total
            ece += (b_total / max(1, total_decided)) * abs(bin_acc - bin_conf)

    return {
        "window": window_size,
        "avg_latency": round(avg_lat, 2),
        "p50": round(p50_lat, 2),
        "p95": round(p95_lat, 2),
        "p99": round(p99_lat, 2),
        "peak_ram_mb": round(peak_ram_mb, 3),
        "accuracy": round(accuracy, 2),
        "brier_score": round(brier_score, 4),
        "ece": round(ece, 4),
        "abstention_rate": round(abstention_rate, 2),
        "regime_switches": regime_switches,
    }


async def main():
    candidate_windows = [500, 1000, 2000, 5000, 10000, 25000, 50000, None]
    print("=" * 95)
    print(" WIN-GO HISTORICAL WINDOW SCALING BENCHMARK (OUT-OF-SAMPLE WALK-FORWARD)")
    print("=" * 95)
    print(f"{'History':<8} | {'Avg Lat':<8} | {'p50':<6} | {'p95':<6} | {'p99':<6} | {'RAM (MB)':<8} | {'Accuracy':<8} | {'Brier':<7} | {'ECE':<6} | {'Abstain %':<9}")
    print("-" * 95)

    results = []
    for w in candidate_windows:
        res = await benchmark_window(w)
        results.append(res)
        w_label = "ADAPTIVE" if w is None else str(w)
        print(
            f"{w_label:<8} | "
            f"{res['avg_latency']:<8.2f} | "
            f"{res['p50']:<6.2f} | "
            f"{res['p95']:<6.2f} | "
            f"{res['p99']:<6.2f} | "
            f"{res['peak_ram_mb']:<8.3f} | "
            f"{res['accuracy']:<8.2f}%| "
            f"{res['brier_score']:<7.4f} | "
            f"{res['ece']:<6.4f} | "
            f"{res['abstention_rate']:<9.2f}%"
        )
    print("=" * 95)


if __name__ == "__main__":
    asyncio.run(main())
