from __future__ import annotations

from dataclasses import dataclass

from features.feature_engine import FeatureBundle

REGIMES = (
    "NORMAL", "CONCENTRATED", "LOW_DIGIT_BIAS", "HIGH_DIGIT_BIAS",
    "DISTRIBUTION_SHIFT", "HIGH_ENTROPY", "LOW_ENTROPY",
    "MODEL_DISAGREEMENT", "UNSTABLE", "UNKNOWN",
)


@dataclass
class RegimeResult:
    regime: str
    detail: dict


class RegimeDetector:
    """Classifies the current statistical regime from live feature data --
    never from a hard-coded assumption about what a pattern "should" mean.
    Everything here is a threshold comparison against measured entropy,
    chi-square shift, and cross-model agreement (agreement is passed in by
    the caller since it depends on the ensemble's own predictions)."""

    def __init__(self, entropy_high: float, entropy_low: float, chi_p_shift: float,
                 model_agreement_unstable: float, min_window_for_regime: int):
        self.entropy_high = entropy_high
        self.entropy_low = entropy_low
        self.chi_p_shift = chi_p_shift
        self.model_agreement_unstable = model_agreement_unstable
        self.min_window_for_regime = min_window_for_regime

    def detect(self, bundle: FeatureBundle, model_std: float) -> RegimeResult:
        if bundle.sample_size < self.min_window_for_regime:
            return RegimeResult("UNKNOWN", {"reason": "insufficient_sample", "sample_size": bundle.sample_size})

        if model_std > self.model_agreement_unstable:
            return RegimeResult("MODEL_DISAGREEMENT", {"model_std": model_std})

        windows_sorted = sorted(bundle.windows.keys())
        if not windows_sorted:
            return RegimeResult("UNKNOWN", {"reason": "no_window_data"})

        largest = windows_sorted[-1]
        smallest_usable = next((w for w in windows_sorted if bundle.windows[w]["n"] >= 50), largest)
        recent_stats = bundle.windows[smallest_usable]
        baseline_stats = bundle.windows[largest]

        entropy = recent_stats["entropy"]
        chi_p = recent_stats["chi_p_value"]

        if chi_p < self.chi_p_shift and smallest_usable != largest:
            # recent window deviates sharply from the historical baseline
            return RegimeResult("DISTRIBUTION_SHIFT", {
                "chi_p_value": chi_p, "window": smallest_usable, "baseline_window": largest,
            })

        if entropy > self.entropy_high:
            return RegimeResult("HIGH_ENTROPY", {"entropy": entropy, "window": smallest_usable})

        if entropy < self.entropy_low:
            deviation = recent_stats["deviation"]
            low_half = sum(deviation[:5])
            high_half = sum(deviation[5:])
            if abs(high_half - low_half) > 0.08:
                regime = "HIGH_DIGIT_BIAS" if high_half > low_half else "LOW_DIGIT_BIAS"
                return RegimeResult(regime, {"entropy": entropy, "low_half": low_half, "high_half": high_half})
            return RegimeResult("CONCENTRATED", {"entropy": entropy})

        return RegimeResult("NORMAL", {"entropy": entropy, "chi_p_value": chi_p})
