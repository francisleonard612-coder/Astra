from __future__ import annotations

from dataclasses import dataclass

from pricing.edge import EdgeResult


@dataclass
class MispricingCheck:
    passes: bool
    reasons_failed: list[str]


def check_mispricing(edge_result: EdgeResult, *, sample_size: int, calibration_score: float,
                      model_agreement: float, minimum_edge: float, minimum_probability: float,
                      minimum_calibration_score: float, minimum_model_agreement: float,
                      minimum_sample_size: int) -> MispricingCheck:
    """Operational definition of "mispricing" per spec section 15: not just a
    nonzero edge, but one backed by enough sample size, calibration quality,
    and model agreement to be credible."""
    reasons: list[str] = []

    if edge_result.edge < minimum_edge:
        reasons.append("insufficient_edge")
    if edge_result.calibrated_probability < minimum_probability:
        reasons.append("probability_below_minimum")
    if calibration_score < minimum_calibration_score:
        reasons.append("poor_calibration")
    if model_agreement < minimum_model_agreement:
        reasons.append("model_disagreement")
    if sample_size < minimum_sample_size:
        reasons.append("insufficient_sample_size")
    if edge_result.expected_value <= 0:
        reasons.append("non_positive_expected_value")

    return MispricingCheck(passes=len(reasons) == 0, reasons_failed=reasons)
