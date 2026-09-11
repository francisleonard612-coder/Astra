from __future__ import annotations

import numpy as np

from models.base import N_DIGITS, normalize


def combine(predictions: dict[str, np.ndarray], weights: dict[str, float | np.ndarray]) -> np.ndarray:
    """Weighted average of model probability vectors.

    `weights[model]` may be a single scalar (one weight applied to that
    model's whole vector -- the original ensemble behavior) or a length-10
    array (a separate weight per digit -- per-digit specialist weighting,
    see learning/online.py). Both are supported transparently: a scalar is
    just broadcast across all 10 digits. Missing/zero weights are skipped;
    the result is always renormalized to sum to 1.
    """
    acc = np.zeros(N_DIGITS)
    total_weight = np.zeros(N_DIGITS)
    for name, vec in predictions.items():
        w = weights.get(name, 0.0)
        w_arr = np.full(N_DIGITS, w, dtype=float) if np.isscalar(w) else np.asarray(w, dtype=float)
        acc += w_arr * vec
        total_weight += w_arr
    if not np.any(total_weight > 0):
        return np.full(N_DIGITS, 1.0 / N_DIGITS)
    # guard any individual digit whose total weight is zero (all contributing
    # models had zero weight for that digit) so division stays well-defined
    safe_total = np.where(total_weight > 0, total_weight, 1.0)
    combined = np.where(total_weight > 0, acc / safe_total, 1.0 / N_DIGITS)
    return normalize(combined)


def model_agreement(predictions: dict[str, np.ndarray], over_barrier: int, under_barrier: int) -> dict[str, float]:
    """Standard deviation across models of P(Over barrier) and P(Under barrier).
    Lower stdev = higher agreement. Returned as a normalized [0, 1]
    "agreement score" (1 = perfect agreement) for both contract sides."""
    over_vals = [float(np.sum(v[over_barrier + 1:])) for v in predictions.values()]
    under_vals = [float(np.sum(v[:under_barrier])) for v in predictions.values()]
    over_std = float(np.std(over_vals)) if len(over_vals) > 1 else 0.0
    under_std = float(np.std(under_vals)) if len(under_vals) > 1 else 0.0
    # a stdev of 0.25 across models on a probability in [0,1] is already very
    # high disagreement; use that as the scale for a 0..1 "agreement" score
    scale = 0.25
    return {
        "over_agreement": max(0.0, 1.0 - min(over_std / scale, 1.0)),
        "under_agreement": max(0.0, 1.0 - min(under_std / scale, 1.0)),
        "over_std": over_std,
        "under_std": under_std,
    }
