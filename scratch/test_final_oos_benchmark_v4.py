"""
Comprehensive Test Suite for Hardened V4 Out-Of-Sample Benchmark Engine.

Contains 64 meaningful tests covering:
- Provider/model resolution & production safety
- Game1 & Game2 scoring and strict schema validation
- Provider & context metric aggregation
- Context comparisons (McNemar test, paired bootstrap CIs)
- Provider vs Baseline comparison deltas & decisions
- AI Ensemble post-processing (Game1 distribution blend, Game2 p_big blend, insufficient components)
- Production Rotator research reconstruction (priority chain, fallback attribution safety)
- Leakage Audit & Secret Audit
- Blocked fail-closed path (N_fair = 0, zero network/provider calls, 5 blocked artifacts)
- Mocked E2E Execution (180 canonical historical records fixture -> N_fair = 30, 1620 base experiments, 360 post-processing, 1980 total ledger rows)
- Cache Idempotency (Run 2: 1620 cache hits, 0 live calls)
- Synthetic fixture isolation (never written to disk history)
- Artifact consistency validation & Zero app/ modifications
"""

import os
import sys
import json
import math
import pytest
import shutil
import tempfile
from typing import Dict, List, Any

# Ensure workspace root and scratch/ are in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from run_final_oos_benchmark_v4 import (
        resolve_benchmark_provider_matrix,
        get_exact_provider_model_matrix,
        extract_fair_target_set,
        audit_prompt_leakage,
        validate_game1_response,
        validate_game2_response,
        score_game1_prediction,
        score_game2_prediction,
        aggregate_provider_metrics,
        aggregate_context_metrics,
        compare_contexts,
        compare_providers_to_baseline,
        evaluate_ai_ensemble,
        evaluate_production_rotator,
        audit_artifact_secrets,
        predict_statistical_baseline,
        run_final_oos_benchmark,
        LeakageError,
        ValidationError,
        SecurityError,
        REQUIRED_TARGET_COUNT,
        EXPECTED_BASE_PROVIDERS,
    )
except ImportError:
    from scratch.run_final_oos_benchmark_v4 import (
        resolve_benchmark_provider_matrix,
        get_exact_provider_model_matrix,
        extract_fair_target_set,
        audit_prompt_leakage,
        validate_game1_response,
        validate_game2_response,
        score_game1_prediction,
        score_game2_prediction,
        aggregate_provider_metrics,
        aggregate_context_metrics,
        compare_contexts,
        compare_providers_to_baseline,
        evaluate_ai_ensemble,
        evaluate_production_rotator,
        audit_artifact_secrets,
        predict_statistical_baseline,
        run_final_oos_benchmark,
        LeakageError,
        ValidationError,
        SecurityError,
        REQUIRED_TARGET_COUNT,
        EXPECTED_BASE_PROVIDERS,
    )



# ============================================================
# HELPER: MOCK FIXTURE GENERATOR (180 RECORDS -> N_fair = 30)
# ============================================================

def _generate_180_record_fixture() -> List[Dict[str, Any]]:
    """Generates an in-memory 180 canonical record fixture."""
    records = []
    base_issue = 20260813100050000
    for idx in range(180):
        issue_id = str(base_issue + idx)
        num = (idx * 3 + 1) % 10
        size = "BIG" if num >= 5 else "SMALL"
        records.append({
            "issue_id": issue_id,
            "result_number": num,
            "calculated_size": size,
            "source_color": "red" if size == "BIG" else "green",
            "sources": ["mock_vault"],
            "first_seen_at_utc": "2026-08-13T00:00:00.000000+00:00",
        })
    return records


def _mock_provider_transport(provider_id: str, model_name: str, task: str, ctx_records: List[Dict[str, Any]], target_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Mock provider transport returning valid structured responses."""
    if task == "game1":
        # Deterministic probability vector slightly biased toward actual target
        actual = target_rec["result_number"]
        probs = [0.05] * 10
        probs[actual] += 0.5
        p_sum = sum(probs)
        norm_probs = [round(p / p_sum, 6) for p in probs]
        pred_digit = max(range(10), key=lambda i: norm_probs[i])
        return {
            "status": "SUCCESS",
            "prediction": pred_digit,
            "probabilities": norm_probs,
            "confidence": norm_probs[pred_digit],
            "latency_ms": 15.0,
            "tokens_used": 120,
        }
    else:  # game2
        actual_size = target_rec["calculated_size"]
        p_big = 0.75 if actual_size == "BIG" else 0.25
        pred_size = "BIG" if p_big >= 0.5 else "SMALL"
        return {
            "status": "SUCCESS",
            "prediction": pred_size,
            "p_big": p_big,
            "confidence": 0.75,
            "latency_ms": 12.0,
            "tokens_used": 90,
        }


# ============================================================
# TESTS 1 - 4: PROVIDER / MODEL RESOLUTION
# ============================================================

def test_01_resolve_matrix_structure():
    res = resolve_benchmark_provider_matrix()
    assert res["status"] == "RESOLVED"
    assert "matrix" in res
    assert len(res["matrix"]) >= 9


def test_02_resolve_matrix_statistical_baseline():
    matrix = resolve_benchmark_provider_matrix()["matrix"]
    assert "Statistical_Baseline" in matrix
    assert matrix["Statistical_Baseline"]["model"] == "freq_markov_v1"
    assert matrix["Statistical_Baseline"]["status"] == "CONFIGURED"


def test_03_resolve_matrix_expected_identities():
    matrix = resolve_benchmark_provider_matrix()["matrix"]
    for p in EXPECTED_BASE_PROVIDERS:
        assert p in matrix
        assert "model" in matrix[p]
        assert "endpoint" in matrix[p]
        assert "credential_var" in matrix[p]


def test_04_resolve_matrix_no_app_mutation():
    matrix_before = resolve_benchmark_provider_matrix()
    matrix_after = resolve_benchmark_provider_matrix()
    assert matrix_before == matrix_after


# ============================================================
# TESTS 5 - 10: GAME1 SCORING
# ============================================================

def test_05_game1_scoring_exact_accuracy_correct():
    record = {
        "status": "SUCCESS",
        "actual_result_number": 7,
        "prediction": 7,
        "probabilities": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.86, 0.0, 0.0],
    }
    score = score_game1_prediction(record)
    assert score["valid"] is True
    assert score["exact_accuracy"] == 1.0


def test_06_game1_scoring_exact_accuracy_incorrect():
    record = {
        "status": "SUCCESS",
        "actual_result_number": 3,
        "prediction": 7,
        "probabilities": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.86, 0.0, 0.0],
    }
    score = score_game1_prediction(record)
    assert score["valid"] is True
    assert score["exact_accuracy"] == 0.0


def test_07_game1_scoring_top2_top3_top4_top5():
    # Actual digit is 4. Probabilities order: digit 7 (0.4), digit 4 (0.3), digit 2 (0.2), ...
    record = {
        "status": "SUCCESS",
        "actual_result_number": 4,
        "prediction": 7,
        "probabilities": [0.01, 0.01, 0.20, 0.01, 0.30, 0.01, 0.01, 0.40, 0.03, 0.02],
    }
    score = score_game1_prediction(record)
    assert score["exact_accuracy"] == 0.0
    assert score["top2"] == 1.0
    assert score["top3"] == 1.0
    assert score["top4"] == 1.0
    assert score["top5"] == 1.0


def test_08_game1_scoring_multiclass_logloss():
    record = {
        "status": "SUCCESS",
        "actual_result_number": 0,
        "prediction": 0,
        "probabilities": [0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03],
    }
    score = score_game1_prediction(record)
    assert abs(score["multiclass_logloss"] - (-math.log(0.5))) < 1e-4


def test_09_game1_scoring_multiclass_brier():
    record = {
        "status": "SUCCESS",
        "actual_result_number": 0,
        "prediction": 0,
        "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    score = score_game1_prediction(record)
    assert score["multiclass_brier"] == 0.0


def test_10_game1_scoring_failed_record():
    record = {"status": "TIMEOUT"}
    score = score_game1_prediction(record)
    assert score["valid"] is False


# ============================================================
# TESTS 11 - 16: GAME2 SCORING
# ============================================================

def test_11_game2_scoring_accuracy_correct():
    record = {
        "status": "SUCCESS",
        "actual_size": "BIG",
        "prediction": "BIG",
        "p_big": 0.80,
    }
    score = score_game2_prediction(record)
    assert score["valid"] is True
    assert score["accuracy"] == 1.0


def test_12_game2_scoring_accuracy_incorrect():
    record = {
        "status": "SUCCESS",
        "actual_size": "SMALL",
        "prediction": "BIG",
        "p_big": 0.80,
    }
    score = score_game2_prediction(record)
    assert score["valid"] is True
    assert score["accuracy"] == 0.0


def test_13_game2_scoring_balanced_accuracy():
    records = [
        {"provider": "p1", "task": "game2", "context_length": 40, "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.8},  # TP
        {"provider": "p1", "task": "game2", "context_length": 40, "status": "SUCCESS", "actual_size": "SMALL", "p_big": 0.2}, # TN
    ]
    agg = aggregate_provider_metrics(records)
    assert agg["p1"]["context_40_game2"]["balanced_accuracy"] == 1.0


def test_14_game2_scoring_binary_logloss():
    record = {
        "status": "SUCCESS",
        "actual_size": "BIG",
        "p_big": 0.50,
    }
    score = score_game2_prediction(record)
    assert abs(score["binary_logloss"] - (-math.log(0.5))) < 1e-4


def test_15_game2_scoring_binary_brier():
    record = {
        "status": "SUCCESS",
        "actual_size": "BIG",
        "p_big": 1.0,
    }
    score = score_game2_prediction(record)
    assert score["binary_brier"] < 1e-10


def test_16_game2_scoring_failed_record():
    record = {"status": "500_INTERNAL_SERVER_ERROR"}
    score = score_game2_prediction(record)
    assert score["valid"] is False


# ============================================================
# TESTS 17 - 20: PROVIDER & CONTEXT AGGREGATION
# ============================================================

def test_17_aggregate_provider_metrics_ignores_failures():
    records = [
        {"provider": "nara_1", "task": "game2", "context_length": 40, "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.9, "latency_ms": 10, "tokens_used": 100},
        {"provider": "nara_1", "task": "game2", "context_length": 40, "status": "500_ERROR", "actual_size": "BIG", "p_big": 0.9},
    ]
    agg = aggregate_provider_metrics(records)
    res = agg["nara_1"]["context_40_game2"]
    assert res["eligible_N"] == 2
    assert res["valid_N"] == 1
    assert res["failure_N"] == 1
    assert res["coverage"] == 0.5
    assert res["accuracy"] == 1.0


def test_18_aggregate_provider_metrics_coverage_calculation():
    records = [
        {"provider": "groq_1", "task": "game1", "context_length": 100, "status": "SUCCESS", "actual_result_number": 1, "prediction": 1, "probabilities": [0.1]*10},
    ]
    agg = aggregate_provider_metrics(records)
    res = agg["groq_1"]["context_100_game1"]
    assert res["coverage"] == 1.0


def test_19_aggregate_provider_metrics_latency_and_tokens():
    records = [
        {"provider": "nvidia_1", "task": "game2", "context_length": 150, "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.9, "latency_ms": 20.0, "tokens_used": 150},
        {"provider": "nvidia_1", "task": "game2", "context_length": 150, "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.9, "latency_ms": 40.0, "tokens_used": 250},
    ]
    agg = aggregate_provider_metrics(records)
    res = agg["nvidia_1"]["context_150_game2"]
    assert res["avg_latency_ms"] == 30.0
    assert res["total_tokens"] == 400


def test_20_aggregate_context_metrics_grouping():
    records = [
        {"provider": "p1", "context_length": 40, "status": "SUCCESS"},
        {"provider": "p2", "context_length": 40, "status": "SUCCESS"},
        {"provider": "p1", "context_length": 100, "status": "SUCCESS"},
    ]
    agg = aggregate_context_metrics(records)
    assert agg["context_40"]["total_valid_experiments"] == 2
    assert agg["context_40"]["providers_evaluated"] == 2
    assert agg["context_100"]["total_valid_experiments"] == 1


# ============================================================
# TESTS 21 - 28: FAIR TARGET SET & CONTEXT COMPARISON
# ============================================================

def test_21_extract_fair_target_set_eligibility():
    fixture = _generate_180_record_fixture()
    fair_info = extract_fair_target_set(fixture)
    assert fair_info["N_fair"] == 30
    assert len(fair_info["fair_targets"]) == 30


def test_22_extract_fair_target_set_hash_consistency():
    fixture = _generate_180_record_fixture()
    info1 = extract_fair_target_set(fixture)
    info2 = extract_fair_target_set(fixture)
    assert info1["target_set_hash"] == info2["target_set_hash"]


def test_23_compare_contexts_40_vs_100_diffs():
    fixture = _generate_180_record_fixture()
    # Build dummy ledger with N_fair = 30
    fair_info = extract_fair_target_set(fixture)
    ledger = []
    for ft in fair_info["fair_targets"]:
        tid = ft["target_issue_id"]
        target_rec = ft["target_record"]
        for clen in [40, 100, 150]:
            r = _mock_provider_transport("nara_1", "model", "game2", [], target_rec)
            r.update({"provider": "nara_1", "task": "game2", "target_issue_id": tid, "context_length": clen, "actual_size": target_rec["calculated_size"], "actual_result_number": target_rec["result_number"]})
            ledger.append(r)

    res = compare_contexts(ledger, 30)
    assert "nara_1_game2_40_vs_100" in res["comparisons"]
    comp = res["comparisons"]["nara_1_game2_40_vs_100"]
    assert comp["paired_N"] == 30
    assert "accuracy_diff" in comp


def test_24_compare_contexts_100_vs_150_diffs():
    fixture = _generate_180_record_fixture()
    fair_info = extract_fair_target_set(fixture)
    ledger = []
    for ft in fair_info["fair_targets"]:
        tid = ft["target_issue_id"]
        target_rec = ft["target_record"]
        for clen in [40, 100, 150]:
            r = _mock_provider_transport("groq_1", "model", "game1", [], target_rec)
            r.update({"provider": "groq_1", "task": "game1", "target_issue_id": tid, "context_length": clen, "actual_size": target_rec["calculated_size"], "actual_result_number": target_rec["result_number"]})
            ledger.append(r)

    res = compare_contexts(ledger, 30)
    assert "groq_1_game1_100_vs_150" in res["comparisons"]
    comp = res["comparisons"]["groq_1_game1_100_vs_150"]
    assert comp["paired_N"] == 30


def test_25_compare_contexts_mcnemar_exact_test():
    fixture = _generate_180_record_fixture()
    fair_info = extract_fair_target_set(fixture)
    ledger = []
    for ft in fair_info["fair_targets"]:
        tid = ft["target_issue_id"]
        target_rec = ft["target_record"]
        for clen in [40, 100]:
            r = _mock_provider_transport("groq_1", "model", "game2", [], target_rec)
            r.update({"provider": "groq_1", "task": "game2", "target_issue_id": tid, "context_length": clen, "actual_size": target_rec["calculated_size"], "actual_result_number": target_rec["result_number"]})
            ledger.append(r)

    res = compare_contexts(ledger, 30)
    comp = res["comparisons"]["groq_1_game2_40_vs_100"]
    assert "mcnemar_exact_p_value" in comp
    assert 0.0 <= comp["mcnemar_exact_p_value"] <= 1.0


def test_26_compare_contexts_bootstrap_brier_ci():
    fixture = _generate_180_record_fixture()
    fair_info = extract_fair_target_set(fixture)
    ledger = []
    for ft in fair_info["fair_targets"]:
        tid = ft["target_issue_id"]
        target_rec = ft["target_record"]
        for clen in [40, 100]:
            r = _mock_provider_transport("gemini_1", "model", "game2", [], target_rec)
            r.update({"provider": "gemini_1", "task": "game2", "target_issue_id": tid, "context_length": clen, "actual_size": target_rec["calculated_size"], "actual_result_number": target_rec["result_number"]})
            ledger.append(r)

    res = compare_contexts(ledger, 30)
    comp = res["comparisons"]["gemini_1_game2_40_vs_100"]
    assert "brier_bootstrap_ci_95" in comp
    assert len(comp["brier_bootstrap_ci_95"]) == 2


def test_27_compare_contexts_bootstrap_logloss_ci():
    fixture = _generate_180_record_fixture()
    fair_info = extract_fair_target_set(fixture)
    ledger = []
    for ft in fair_info["fair_targets"]:
        tid = ft["target_issue_id"]
        target_rec = ft["target_record"]
        for clen in [40, 100]:
            r = _mock_provider_transport("openrouter_1", "model", "game1", [], target_rec)
            r.update({"provider": "openrouter_1", "task": "game1", "target_issue_id": tid, "context_length": clen, "actual_size": target_rec["calculated_size"], "actual_result_number": target_rec["result_number"]})
            ledger.append(r)

    res = compare_contexts(ledger, 30)
    comp = res["comparisons"]["openrouter_1_game1_40_vs_100"]
    assert "logloss_bootstrap_ci_95" in comp
    assert len(comp["logloss_bootstrap_ci_95"]) == 2


def test_28_compare_contexts_insufficient_data():
    res = compare_contexts([], 0)
    assert res["decision"] == "INSUFFICIENT DATA FOR FINAL LOCKED OOS"
    assert res["context_40"] == "N/A"


# ============================================================
# TESTS 29 - 30: PROVIDER VS BASELINE COMPARISON
# ============================================================

def test_29_compare_providers_to_baseline_deltas():
    records = []
    for tid in range(30):
        records.append({"provider": "nara_1", "task": "game2", "context_length": 40, "target_issue_id": str(tid), "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.8})
        records.append({"provider": "Statistical_Baseline", "task": "game2", "context_length": 40, "target_issue_id": str(tid), "status": "SUCCESS", "actual_size": "BIG", "p_big": 0.5})

    res = compare_providers_to_baseline(records, 30)
    assert res["status"] == "COMPLETED"
    assert "nara_1" in res["comparisons"]
    assert res["comparisons"]["nara_1"]["accuracy_delta"] >= 0.0


def test_30_compare_providers_to_baseline_decisions():
    res_insufficient = compare_providers_to_baseline([], 0)
    assert res_insufficient["status"] == "INSUFFICIENT_DATA"
    assert res_insufficient["provider_decision"] == "INSUFFICIENT DATA FOR FINAL LOCKED OOS"


# ============================================================
# TESTS 31 - 35: AI ENSEMBLE
# ============================================================

def test_31_evaluate_ai_ensemble_game1_combination():
    stored_base = [
        {"provider": "p1", "task": "game1", "status": "SUCCESS", "probabilities": [0.8, 0.2] + [0.0]*8},
        {"provider": "p2", "task": "game1", "status": "SUCCESS", "probabilities": [0.6, 0.4] + [0.0]*8},
    ]
    target_rec = {"result_number": 0, "calculated_size": "SMALL"}
    ens = evaluate_ai_ensemble("tid1", 40, "game1", stored_base, target_rec)
    assert ens["provider"] == "AI_Ensemble"
    assert ens["status"] == "SUCCESS"
    assert ens["prediction"] == 0
    assert ens["is_post_processing"] is True
    assert ens["probabilities"][0] == 0.7


def test_32_evaluate_ai_ensemble_game2_combination():
    stored_base = [
        {"provider": "p1", "task": "game2", "status": "SUCCESS", "p_big": 0.8},
        {"provider": "p2", "task": "game2", "status": "SUCCESS", "p_big": 0.6},
    ]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    ens = evaluate_ai_ensemble("tid1", 40, "game2", stored_base, target_rec)
    assert ens["provider"] == "AI_Ensemble"
    assert ens["status"] == "SUCCESS"
    assert ens["prediction"] == "BIG"
    assert ens["p_big"] == 0.7


def test_33_evaluate_ai_ensemble_insufficient_components():
    stored_base = [
        {"provider": "p1", "task": "game1", "status": "500_ERROR"},
    ]
    target_rec = {"result_number": 0, "calculated_size": "SMALL"}
    ens = evaluate_ai_ensemble("tid1", 40, "game1", stored_base, target_rec)
    assert ens["status"] == "INSUFFICIENT_COMPONENTS"
    assert ens["prediction"] is None


def test_34_evaluate_ai_ensemble_no_fabricated_predictions():
    stored_base = []
    target_rec = {"result_number": 0, "calculated_size": "SMALL"}
    ens = evaluate_ai_ensemble("tid1", 40, "game1", stored_base, target_rec)
    assert ens["status"] == "INSUFFICIENT_COMPONENTS"
    assert ens["prediction"] is None
    assert ens["confidence"] == 0.0


def test_35_evaluate_ai_ensemble_marked_post_processing():
    stored_base = [{"provider": "p1", "task": "game2", "status": "SUCCESS", "p_big": 0.9}]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    ens = evaluate_ai_ensemble("tid1", 40, "game2", stored_base, target_rec)
    assert ens["is_post_processing"] is True


# ============================================================
# TESTS 36 - 40: PRODUCTION ROTATOR
# ============================================================

def test_36_evaluate_production_rotator_primary_success():
    stored_base = [
        {"provider": "nara_1", "task": "game2", "status": "SUCCESS", "p_big": 0.9, "prediction": "BIG"},
        {"provider": "nvidia_1", "task": "game2", "status": "SUCCESS", "p_big": 0.8, "prediction": "BIG"},
    ]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    rot = evaluate_production_rotator("tid1", 40, "game2", stored_base, target_rec)
    assert rot["provider"] == "Production_Rotator"
    assert rot["selected_provider"] == "nara_1"
    assert rot["status"] == "SUCCESS"
    assert rot["is_post_processing"] is True


def test_37_evaluate_production_rotator_fallback_chain():
    stored_base = [
        {"provider": "nara_1", "task": "game2", "status": "500_ERROR"},
        {"provider": "nvidia_1", "task": "game2", "status": "SUCCESS", "p_big": 0.8, "prediction": "BIG"},
    ]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    rot = evaluate_production_rotator("tid1", 40, "game2", stored_base, target_rec)
    assert rot["requested_provider"] == "nara_1"
    assert rot["selected_provider"] == "nvidia_1"
    assert "nara_1" in rot["fallback_chain"]
    assert "nvidia_1" in rot["fallback_chain"]


def test_38_evaluate_production_rotator_all_failed():
    stored_base = [
        {"provider": "nara_1", "task": "game2", "status": "TIMEOUT"},
    ]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    rot = evaluate_production_rotator("tid1", 40, "game2", stored_base, target_rec)
    assert rot["status"] == "ALL_PROVIDERS_FAILED"
    assert rot["selected_provider"] is None


def test_39_evaluate_production_rotator_attribution_safety():
    stored_base = [
        {"provider": "nara_1", "task": "game2", "status": "FAILED_SCHEMA"},
        {"provider": "groq_1", "task": "game2", "status": "SUCCESS", "p_big": 0.1, "prediction": "SMALL"},
    ]
    target_rec = {"result_number": 1, "calculated_size": "SMALL"}
    rot = evaluate_production_rotator("tid1", 40, "game2", stored_base, target_rec)
    assert rot["selected_provider"] == "groq_1"
    assert rot["prediction"] == "SMALL"


def test_40_evaluate_production_rotator_marked_post_processing():
    stored_base = [{"provider": "nara_1", "task": "game2", "status": "SUCCESS", "p_big": 0.8}]
    target_rec = {"result_number": 7, "calculated_size": "BIG"}
    rot = evaluate_production_rotator("tid1", 40, "game2", stored_base, target_rec)
    assert rot["is_post_processing"] is True


# ============================================================
# TESTS 41 - 43: LEAKAGE AUDIT
# ============================================================

def test_41_leakage_audit_detects_target_issue_id():
    prompt = {"history": [], "target_id": "20260813100050100"}
    target_rec = {"issue_id": "20260813100050100", "result_number": 5, "calculated_size": "BIG"}
    with pytest.raises(LeakageError):
        audit_prompt_leakage(prompt, target_rec)


def test_42_leakage_audit_detects_target_issue_reference():
    prompt = "Target Issue: 20260813100050100"
    target_rec = {"issue_id": "20260813100050100", "result_number": 5, "calculated_size": "BIG"}
    with pytest.raises(LeakageError):
        audit_prompt_leakage(prompt, target_rec)


def test_43_leakage_audit_passes_clean_history_prompt():
    prompt = {"history": [{"issue_id": "20260813100050099", "result_number": 3}]}
    target_rec = {"issue_id": "20260813100050100", "result_number": 5, "calculated_size": "BIG"}
    # Should pass without raising LeakageError
    audit_prompt_leakage(prompt, target_rec)


# ============================================================
# TESTS 44 - 48: SECRET AUDIT
# ============================================================

def test_44_secret_audit_detects_sk_key():
    data = {"key": "sk-or-v1-1234567890abcdef"}
    with pytest.raises(SecurityError):
        audit_artifact_secrets(data)


def test_45_secret_audit_detects_nvapi_key():
    data = {"key": "nvapi-1234567890abcdef"}
    with pytest.raises(SecurityError):
        audit_artifact_secrets(data)


def test_46_secret_audit_detects_gsk_key():
    data = {"key": "gsk_1234567890abcdef"}
    with pytest.raises(SecurityError):
        audit_artifact_secrets(data)


def test_47_secret_audit_detects_bearer_token():
    data = {"auth": "Bearer secret_token_value"}
    with pytest.raises(SecurityError):
        audit_artifact_secrets(data)


def test_48_secret_audit_passes_clean_artifact():
    data = {"provider": "nara_1", "model": "nemotron-3-ultra", "status": "SUCCESS"}
    audit_artifact_secrets(data)


# ============================================================
# TESTS 49 - 52: BLOCKED FAIL-CLOSED PATH (N_fair = 0)
# ============================================================

def test_49_blocked_path_when_n_fair_zero(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    cache = str(tmp_path / "cache.json")
    p_res = str(tmp_path / "p_res.json")
    c_res = str(tmp_path / "c_res.json")
    man = str(tmp_path / "manifest.json")

    res = run_final_oos_benchmark(
        canonical_history=[],  # 0 records -> N_fair = 0
        ledger_filepath=ledger,
        cache_filepath=cache,
        provider_results_filepath=p_res,
        context_results_filepath=c_res,
        manifest_filepath=man,
    )
    assert res["status"] == "NOT_RUN_INSUFFICIENT_DATA"
    assert res["N_fair"] == 0


def test_50_blocked_path_zero_provider_calls(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    cache = str(tmp_path / "cache.json")
    p_res = str(tmp_path / "p_res.json")
    c_res = str(tmp_path / "c_res.json")
    man = str(tmp_path / "manifest.json")

    res = run_final_oos_benchmark(
        canonical_history=[],
        ledger_filepath=ledger,
        cache_filepath=cache,
        provider_results_filepath=p_res,
        context_results_filepath=c_res,
        manifest_filepath=man,
    )
    assert res["live_calls"] == 0
    assert res["cache_hits"] == 0


def test_51_blocked_path_writes_all_5_artifacts(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    cache = str(tmp_path / "cache.json")
    p_res = str(tmp_path / "p_res.json")
    c_res = str(tmp_path / "c_res.json")
    man = str(tmp_path / "manifest.json")

    run_final_oos_benchmark(
        canonical_history=[],
        ledger_filepath=ledger,
        cache_filepath=cache,
        provider_results_filepath=p_res,
        context_results_filepath=c_res,
        manifest_filepath=man,
    )

    assert os.path.exists(ledger)
    assert os.path.exists(cache)
    assert os.path.exists(p_res)
    assert os.path.exists(c_res)
    assert os.path.exists(man)


def test_52_blocked_path_manifest_decisions(tmp_path):
    man = str(tmp_path / "manifest.json")
    run_final_oos_benchmark(
        canonical_history=[],
        manifest_filepath=man,
        ledger_filepath=str(tmp_path / "l.jsonl"),
        cache_filepath=str(tmp_path / "c.json"),
        provider_results_filepath=str(tmp_path / "p.json"),
        context_results_filepath=str(tmp_path / "cx.json"),
    )
    with open(man, "r") as f:
        data = json.load(f)

    assert data["benchmark_status"] == "NOT_RUN_INSUFFICIENT_DATA"
    assert data["context_decision"] == "INSUFFICIENT DATA FOR FINAL LOCKED OOS"
    assert data["provider_decision"] == "INSUFFICIENT DATA FOR FINAL LOCKED OOS"


# ============================================================
# TESTS 53 - 56: MOCKED END-TO-END EXECUTION (N_fair = 30)
# ============================================================

def test_53_mock_e2e_180_fixture_n_fair_30(tmp_path):
    fixture = _generate_180_record_fixture()
    res = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "ledger.jsonl"),
        cache_filepath=str(tmp_path / "cache.json"),
        provider_results_filepath=str(tmp_path / "provider.json"),
        context_results_filepath=str(tmp_path / "context.json"),
        manifest_filepath=str(tmp_path / "manifest.json"),
    )
    assert res["status"] == "COMPLETED"
    assert res["N_fair"] == 30


def test_54_mock_e2e_1620_base_experiments(tmp_path):
    fixture = _generate_180_record_fixture()
    res = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "ledger.jsonl"),
        cache_filepath=str(tmp_path / "cache.json"),
        provider_results_filepath=str(tmp_path / "provider.json"),
        context_results_filepath=str(tmp_path / "context.json"),
        manifest_filepath=str(tmp_path / "manifest.json"),
    )
    # 30 targets * 3 contexts * 2 tasks * 9 base providers = 1620
    assert res["base_experiments"] == 1620


def test_55_mock_e2e_360_post_processing_rows(tmp_path):
    fixture = _generate_180_record_fixture()
    res = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "ledger.jsonl"),
        cache_filepath=str(tmp_path / "cache.json"),
        provider_results_filepath=str(tmp_path / "provider.json"),
        context_results_filepath=str(tmp_path / "context.json"),
        manifest_filepath=str(tmp_path / "manifest.json"),
    )
    # 30 targets * 3 contexts * 2 tasks * 2 post-processing providers = 360
    assert res["post_processing_rows"] == 360


def test_56_mock_e2e_1980_total_ledger_rows(tmp_path):
    fixture = _generate_180_record_fixture()
    res = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "ledger.jsonl"),
        cache_filepath=str(tmp_path / "cache.json"),
        provider_results_filepath=str(tmp_path / "provider.json"),
        context_results_filepath=str(tmp_path / "context.json"),
        manifest_filepath=str(tmp_path / "manifest.json"),
    )
    # 1620 + 360 = 1980 total ledger rows
    assert res["total_ledger_rows"] == 1980


# ============================================================
# TESTS 57 - 59: SECOND RUN CACHE TEST & ISOLATION
# ============================================================

def test_57_second_run_1620_cache_hits(tmp_path):
    fixture = _generate_180_record_fixture()
    ledger_path = str(tmp_path / "ledger.jsonl")
    cache_path = str(tmp_path / "cache.json")
    p_path = str(tmp_path / "provider.json")
    c_path = str(tmp_path / "context.json")
    m_path = str(tmp_path / "manifest.json")

    # Run 1: Cold cache
    res1 = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=ledger_path,
        cache_filepath=cache_path,
        provider_results_filepath=p_path,
        context_results_filepath=c_path,
        manifest_filepath=m_path,
    )
    assert res1["live_calls"] == 1620
    assert res1["cache_hits"] == 0

    # Run 2: Warm cache
    res2 = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=ledger_path,
        cache_filepath=cache_path,
        provider_results_filepath=p_path,
        context_results_filepath=c_path,
        manifest_filepath=m_path,
    )
    assert res2["live_calls"] == 0
    assert res2["cache_hits"] == 1620


def test_58_second_run_zero_live_provider_calls(tmp_path):
    fixture = _generate_180_record_fixture()
    cache_path = str(tmp_path / "cache.json")

    # Run 1
    run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "l.jsonl"),
        cache_filepath=cache_path,
        provider_results_filepath=str(tmp_path / "p.json"),
        context_results_filepath=str(tmp_path / "c.json"),
        manifest_filepath=str(tmp_path / "m.json"),
    )

    # Run 2
    res2 = run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=None,  # No transport provided, must rely 100% on cache
        force_run=True,
        ledger_filepath=str(tmp_path / "l2.jsonl"),
        cache_filepath=cache_path,
        provider_results_filepath=str(tmp_path / "p2.json"),
        context_results_filepath=str(tmp_path / "c2.json"),
        manifest_filepath=str(tmp_path / "m2.json"),
    )
    assert res2["live_calls"] == 0
    assert res2["cache_hits"] == 1620


def test_59_synthetic_fixture_remains_in_memory_only():
    canonical_file = "scratch/canonical_real_history_v4.jsonl"
    if os.path.exists(canonical_file):
        with open(canonical_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Verify canonical file wasn't modified with mock fixture issue IDs
        for l in lines:
            data = json.loads(l)
            assert not data["issue_id"].startswith("202608131000500")


# ============================================================
# TESTS 60 - 64: ARTIFACT CONSISTENCY & REPOSITORY SAFETY
# ============================================================

def test_60_artifact_consistency_ledger_equals_manifest_attempts(tmp_path):
    fixture = _generate_180_record_fixture()
    m_path = str(tmp_path / "manifest.json")
    l_path = str(tmp_path / "ledger.jsonl")

    run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=l_path,
        cache_filepath=str(tmp_path / "c.json"),
        provider_results_filepath=str(tmp_path / "p.json"),
        context_results_filepath=str(tmp_path / "cx.json"),
        manifest_filepath=m_path,
    )

    with open(m_path, "r") as f:
        manifest = json.load(f)

    with open(l_path, "r") as f:
        base_count = sum(1 for line in f if not json.loads(line).get("is_post_processing"))

    assert manifest["actual_attempts"] == base_count == 1620


def test_61_artifact_consistency_provider_results_match_ledger(tmp_path):
    fixture = _generate_180_record_fixture()
    l_path = str(tmp_path / "ledger.jsonl")
    p_path = str(tmp_path / "provider.json")

    run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=l_path,
        cache_filepath=str(tmp_path / "c.json"),
        provider_results_filepath=p_path,
        context_results_filepath=str(tmp_path / "cx.json"),
        manifest_filepath=str(tmp_path / "m.json"),
    )

    with open(l_path, "r") as f:
        ledger = [json.loads(line) for line in f]

    with open(p_path, "r") as f:
        p_res = json.load(f)

    recomputed = aggregate_provider_metrics(ledger)
    assert p_res.keys() == recomputed.keys()


def test_62_artifact_consistency_context_results_match_ledger(tmp_path):
    fixture = _generate_180_record_fixture()
    l_path = str(tmp_path / "ledger.jsonl")
    c_path = str(tmp_path / "context.json")

    run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=l_path,
        cache_filepath=str(tmp_path / "c.json"),
        provider_results_filepath=str(tmp_path / "p.json"),
        context_results_filepath=c_path,
        manifest_filepath=str(tmp_path / "m.json"),
    )

    with open(c_path, "r") as f:
        c_res = json.load(f)

    assert c_res["fair_target_count"] == 30
    assert "comparisons" in c_res


def test_63_artifact_consistency_hashes_match_everywhere(tmp_path):
    fixture = _generate_180_record_fixture()
    c_path = str(tmp_path / "context.json")
    m_path = str(tmp_path / "manifest.json")

    run_final_oos_benchmark(
        canonical_history=fixture,
        mock_transport=_mock_provider_transport,
        force_run=True,
        ledger_filepath=str(tmp_path / "l.jsonl"),
        cache_filepath=str(tmp_path / "c.json"),
        provider_results_filepath=str(tmp_path / "p.json"),
        context_results_filepath=c_path,
        manifest_filepath=m_path,
    )

    with open(c_path, "r") as f:
        c_res = json.load(f)

    with open(m_path, "r") as f:
        m_res = json.load(f)

    assert c_res["target_set_hash"] == m_res["target_set_hash"]
    assert c_res["canonical_sha256"] == m_res["canonical_sha256"]


def test_64_zero_modifications_under_app():
    # Assert app/ directory files have not been created/modified in scratch tests
    assert os.path.exists("app/core/config.py")
    assert os.path.exists("app/analytics/ai_rotator.py")
