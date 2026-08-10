"""
Test Suite for Evidence Classifier & Accuracy Claim Guard.

Verifies:
1. EXPERIMENTAL classification when N < 100 or accuracy < 52%.
2. PROMISING classification when 100 <= N < 500.
3. VALIDATED classification when N >= 500 and p-value < 0.05.
4. PRODUCTION_VALIDATED classification requires N >= 1,000 and live production flag.
5. NOT_VALIDATED classification for non-significant edge.
"""

from app.analytics.evidence_classifier import classify_accuracy_evidence


def test_classify_evidence_levels():
    # 0 samples -> NOT_VALIDATED
    c0 = classify_accuracy_evidence(0, 0)
    assert c0.level == "NOT_VALIDATED"

    # Small sample (N=50) -> EXPERIMENTAL
    c1 = classify_accuracy_evidence(30, 50)
    assert c1.level == "EXPERIMENTAL"

    # Moderate sample (N=200, 58%) -> PROMISING
    c2 = classify_accuracy_evidence(116, 200)
    assert c2.level == "PROMISING"

    # Large sample (N=600, 58.5%, p < 0.05) -> VALIDATED
    c3 = classify_accuracy_evidence(351, 600)
    assert c3.level == "VALIDATED"

    # Large production sample (N=1000, 58.6%, live=True) -> PRODUCTION_VALIDATED
    c4 = classify_accuracy_evidence(586, 1000, is_live_production=True)
    assert c4.level == "PRODUCTION_VALIDATED"

    # Large local sample without live DB flag -> VALIDATED (not PRODUCTION_VALIDATED)
    c5 = classify_accuracy_evidence(586, 1000, is_live_production=False)
    assert c5.level == "VALIDATED"
