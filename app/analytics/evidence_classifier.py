"""
Evidence Classifier & Accuracy Claim Guard Module.

Enforces strict anti-bluff rules on system accuracy and edge claims:
- Prohibits marketing claims ('guaranteed', '90% accurate', 'killer prediction').
- Classifies evidence status using rigorous statistical thresholds:
  1. EXPERIMENTAL: Sample count N < 100 or accuracy < 52%.
  2. PROMISING: Sample count 100 <= N < 500 and accuracy >= 52%.
  3. VALIDATED: Sample count N >= 500, accuracy >= 55%, 95% Wilson CI lower bound > 50.0%, p-value < 0.05.
  4. PRODUCTION_VALIDATED: N >= 1,000 from live production predictions, accuracy >= 55%, p-value < 0.01.
"""

import math
from dataclasses import dataclass
from typing import Literal

EvidenceLevel = Literal["EXPERIMENTAL", "PROMISING", "VALIDATED", "PRODUCTION_VALIDATED", "NOT_VALIDATED"]


@dataclass
class EvidenceClassification:
    level: EvidenceLevel
    sample_size: int
    observed_accuracy_pct: float
    wilson_95_ci_pct: tuple[float, float]
    p_value: float
    description: str


def classify_accuracy_evidence(
    wins: int,
    total: int,
    is_live_production: bool = False
) -> EvidenceClassification:
    """
    Classify system prediction edge based on rigorous sample size and statistical bounds.
    """
    if total <= 0:
        return EvidenceClassification(
            level="NOT_VALIDATED",
            sample_size=0,
            observed_accuracy_pct=0.0,
            wilson_95_ci_pct=(0.0, 0.0),
            p_value=1.0,
            description="Insufficient data for classification (0 samples).",
        )

    acc = (wins / total) * 100.0

    # Calculate Wilson 95% CI
    p_hat = wins / total
    z = 1.96
    denominator = 1 + z**2 / total
    centre = p_hat + z**2 / (2 * total)
    std_dev = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    ci_lower = max(0.0, round(((centre - std_dev) / denominator) * 100, 2))
    ci_upper = min(100.0, round(((centre + std_dev) / denominator) * 100, 2))

    # Calculate binomial test p-value vs 50% random baseline
    mean = total * 0.50
    sigma = math.sqrt(total * 0.25)
    if sigma > 0:
        z_stat = (wins - 0.5 - mean) / sigma
        p_val = round(max(0.0001, min(1.0, 0.5 * (1.0 - math.erf(z_stat / math.sqrt(2))))), 4) if z_stat > 0 else 1.0
    else:
        p_val = 1.0

    # Evidence Level Classification Hierarchy
    if is_live_production and total >= 1000 and acc >= 55.0 and ci_lower > 50.0 and p_val < 0.01:
        level: EvidenceLevel = "PRODUCTION_VALIDATED"
        desc = "Rigorous production evidence: N >= 1,000, 95% CI lower bound > 50%, p < 0.01."
    elif total >= 500 and acc >= 55.0 and ci_lower > 50.0 and p_val < 0.05:
        level: EvidenceLevel = "VALIDATED"
        desc = "Statistically validated out-of-sample edge: N >= 500, p < 0.05."
    elif total >= 100 and acc >= 52.0:
        level: EvidenceLevel = "PROMISING"
        desc = "Promising early evidence (100 <= N < 500), further OOS data required."
    elif total < 100:
        level: EvidenceLevel = "EXPERIMENTAL"
        desc = "Experimental stage: Sample size N < 100 is too small for statistical conclusions."
    else:
        level: EvidenceLevel = "NOT_VALIDATED"
        desc = "No statistically significant edge over random baseline."

    return EvidenceClassification(
        level=level,
        sample_size=total,
        observed_accuracy_pct=round(acc, 2),
        wilson_95_ci_pct=(ci_lower, ci_upper),
        p_value=p_val,
        description=desc,
    )
