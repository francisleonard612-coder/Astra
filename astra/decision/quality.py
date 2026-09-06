from __future__ import annotations

from dataclasses import dataclass

from pricing.edge import EdgeResult

# Component weights for the 0-100 quality score. Kept in one place so the
# scoring function is easy to tune/test in isolation (spec section 16 calls
# for this to be configurable and tested).
_WEIGHTS = {
    "edge": 0.30,
    "calibration": 0.20,
    "model_agreement": 0.20,
    "regime_stability": 0.10,
    "sample_size": 0.10,
    "data_quality": 0.10,
}

_REGIME_STABILITY = {
    "NORMAL": 1.0, "CONCENTRATED": 0.6, "LOW_DIGIT_BIAS": 0.6, "HIGH_DIGIT_BIAS": 0.6,
    "HIGH_ENTROPY": 0.5, "LOW_ENTROPY": 0.5, "DISTRIBUTION_SHIFT": 0.2,
    "MODEL_DISAGREEMENT": 0.1, "UNSTABLE": 0.0, "UNKNOWN": 0.0,
}


@dataclass
class TradeQuality:
    score: float
    components: dict[str, float]


def score_trade(edge_result: EdgeResult, *, calibration_score: float, model_agreement: float,
                 regime: str, sample_size: int, target_sample_size: int,
                 data_quality: float = 1.0, max_meaningful_edge: float = 0.15) -> TradeQuality:
    edge_component = min(max(edge_result.edge, 0.0) / max_meaningful_edge, 1.0)
    sample_component = min(sample_size / max(target_sample_size, 1), 1.0)
    regime_component = _REGIME_STABILITY.get(regime, 0.0)

    components = {
        "edge": edge_component,
        "calibration": calibration_score,
        "model_agreement": model_agreement,
        "regime_stability": regime_component,
        "sample_size": sample_component,
        "data_quality": data_quality,
    }
    score = sum(_WEIGHTS[k] * v for k, v in components.items()) * 100.0
    return TradeQuality(score=score, components=components)
