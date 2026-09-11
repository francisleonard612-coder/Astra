"""
Feature engine.

Turns a SymbolState into (a) a numeric feature vector usable by the sklearn
models, and (b) a human-readable feature dict used by the regime detector,
explanation object, and non-ML models.

Design notes:
- Gaps are exposed as raw counters only -- nothing here assumes "long
  absence == due". Whether gap features carry predictive information is an
  empirical question the model layer and its rolling performance answer, not
  something hard-coded in feature engineering.
- High-order windows (2500+) degrade gracefully to whatever history exists;
  no window ever divides by zero.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from state.rolling_state import SymbolState

N_DIGITS = 10


@dataclass
class FeatureBundle:
    symbol: str
    windows: dict[int, dict] = field(default_factory=dict)   # window_size -> per-window stats
    gaps: dict[int, int] = field(default_factory=dict)        # digit -> current gap
    streaks: dict[str, int] = field(default_factory=dict)
    entropy: dict[int, float] = field(default_factory=dict)   # window -> normalized entropy
    sample_size: int = 0
    vector: np.ndarray | None = None  # flattened feature vector for sklearn models


def _digit_counts(seq: list[int]) -> np.ndarray:
    counts = np.zeros(N_DIGITS, dtype=float)
    for d in seq:
        counts[d] += 1
    return counts


def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / math.log(N_DIGITS))  # normalized to [0, 1]


def _chi_square_vs_uniform(counts: np.ndarray) -> tuple[float, float]:
    n = counts.sum()
    if n == 0:
        return 0.0, 1.0
    expected = n / N_DIGITS
    stat = float(np.sum((counts - expected) ** 2 / expected))
    # crude p-value approximation via chi2 survival for df=9 using a
    # lightweight series (avoids a scipy dependency for this one call)
    try:
        from scipy import stats as scipy_stats
        p_value = float(scipy_stats.chi2.sf(stat, df=N_DIGITS - 1))
    except Exception:  # noqa: BLE001
        p_value = 1.0 if stat < 16.9 else 0.01  # 16.9 ~ chi2 critical value, df=9, alpha=0.05
    return stat, p_value


def build_features(state: SymbolState, windows: list[int]) -> FeatureBundle:
    bundle = FeatureBundle(symbol=state.symbol, sample_size=state.total_observed)

    for w in windows:
        seq = state.window(w)
        if not seq:
            continue
        counts = _digit_counts(seq)
        total = counts.sum()
        freq = counts / total if total else counts
        uniform = np.full(N_DIGITS, 1.0 / N_DIGITS)
        deviation = freq - uniform
        chi_stat, chi_p = _chi_square_vs_uniform(counts)
        bundle.windows[w] = {
            "counts": counts.tolist(),
            "freq": freq.tolist(),
            "deviation": deviation.tolist(),
            "entropy": _shannon_entropy(freq),
            "chi_square": chi_stat,
            "chi_p_value": chi_p,
            "n": int(total),
        }
        bundle.entropy[w] = _shannon_entropy(freq)

    for d in range(N_DIGITS):
        bundle.gaps[d] = state.gap(d)

    bundle.streaks = {
        "same_digit": state.same_digit_streak,
        "high": state.high_streak,
        "low": state.low_streak,
        "over2": state.over2_streak,
        "under7": state.under7_streak,
    }

    bundle.vector = _to_vector(bundle, windows)
    return bundle


def _to_vector(bundle: FeatureBundle, windows: list[int]) -> np.ndarray:
    """Flatten the most decision-relevant features into a fixed-length vector for sklearn models."""
    parts: list[float] = []
    for w in windows:
        stats = bundle.windows.get(w)
        if stats is None:
            parts.extend([0.0] * N_DIGITS)
            parts.append(0.0)
            continue
        parts.extend(stats["freq"])
        parts.append(stats["entropy"])
    for d in range(N_DIGITS):
        parts.append(min(bundle.gaps.get(d, 0), 5000) / 5000.0)  # normalized gap
    parts.extend([
        bundle.streaks.get("same_digit", 0) / 50.0,
        bundle.streaks.get("high", 0) / 50.0,
        bundle.streaks.get("low", 0) / 50.0,
        bundle.streaks.get("over2", 0) / 50.0,
        bundle.streaks.get("under7", 0) / 50.0,
    ])
    return np.array(parts, dtype=float)
