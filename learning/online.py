"""
Per-digit specialist weighting.

The spec (sections 5 & 7) asks for 10 digit specialists that can each
favor a different mix of underlying models -- e.g. digit 0 might be best
called by Bayesian+Markov+XGBoost while digit 4 is best called by
transition+GBM+frequency -- with the final 10-digit vector built by
combining those specialists.

SCOPE NOTE on how this is implemented: this build does NOT instantiate 10
separate trained model objects per digit per symbol (that's 8 models x 10
digits x N symbols independently-fitted objects -- a real memory/compute
cost on a Railway worker running every R_*/1HZ* symbol at once). Instead,
every model still predicts the full 10-digit vector as before (shared
feature representation, as spec section 7 also calls for), and THIS module
gives the ensemble a separate weight per (model, digit) pair, learned from
each model's own rolling per-digit accuracy. The practical effect is the
same thing the spec is after -- "digit 4 ends up mostly listening to
Markov+frequency, digit 0 ends up mostly listening to Bayesian+XGBoost" --
achieved by adaptive weighting of shared models rather than by duplicating
model objects per digit. If you want literal separate fitted specialist
objects per digit instead, that's a bigger change (10x the batch-model
memory footprint) and should be scoped as its own follow-up.

Each (model, digit) pair is scored with its own rolling BINARY log-loss:
treat the model's stated P(digit=d) as a probability estimate for the
binary event "was the realized digit d", scored against y=1/0 accordingly.
This is exactly how you'd score 10 independent binary specialists, whether
or not they're actually separate objects under the hood.
"""
from __future__ import annotations

from collections import deque

import numpy as np

N_DIGITS = 10
_MIN_HISTORY = 30  # per-(model,digit) samples needed before trusting learned weight over the initial default


class PerformanceTracker:
    def __init__(self, initial_weights: dict[str, float], min_weight: float, window: int):
        self.initial_weights = dict(initial_weights)
        self.min_weight = min_weight
        self.window = window
        # per-model overall multiclass log-loss (kept for reporting/repository logging)
        self._logloss_overall: dict[str, deque[float]] = {name: deque(maxlen=window) for name in initial_weights}
        # per-(model, digit) binary log-loss -- this is what per-digit weighting is learned from
        self._logloss_digit: dict[str, list[deque[float]]] = {
            name: [deque(maxlen=window) for _ in range(N_DIGITS)] for name in initial_weights
        }

    def _ensure_model(self, name: str) -> None:
        if name not in self._logloss_overall:
            self._logloss_overall[name] = deque(maxlen=self.window)
        if name not in self._logloss_digit:
            self._logloss_digit[name] = [deque(maxlen=self.window) for _ in range(N_DIGITS)]

    def record(self, predictions: dict[str, np.ndarray], actual_digit: int) -> None:
        for name, vec in predictions.items():
            self._ensure_model(name)

            p_actual = float(np.clip(vec[actual_digit], 1e-9, 1.0))
            self._logloss_overall[name].append(float(-np.log(p_actual)))

            for d in range(N_DIGITS):
                y = 1.0 if d == actual_digit else 0.0
                p = float(np.clip(vec[d], 1e-9, 1 - 1e-9))
                binary_loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
                self._logloss_digit[name][d].append(binary_loss)

    def rolling_log_loss(self, name: str) -> float | None:
        """Overall (multiclass) rolling log-loss for a model -- used for
        reporting/dashboards, not for the per-digit weights themselves."""
        buf = self._logloss_overall.get(name)
        if not buf:
            return None
        return float(np.mean(buf))

    def rolling_log_loss_for_digit(self, name: str, digit: int) -> float | None:
        buf = self._logloss_digit.get(name, [None] * N_DIGITS)[digit]
        if not buf:
            return None
        return float(np.mean(buf))

    def current_weights(self) -> dict[str, np.ndarray]:
        """Returns, per model, a length-10 array of weights -- one per digit.
        For each digit independently: models with lower rolling binary
        log-loss on THAT digit get more weight; weights are normalized to
        sum to 1 across models for each digit, with a floor so no model is
        ever driven to exactly zero on any digit (it might recover)."""
        per_digit_weights: dict[str, np.ndarray] = {name: np.zeros(N_DIGITS) for name in self.initial_weights}

        for d in range(N_DIGITS):
            raw = {}
            for name, initial in self.initial_weights.items():
                buf = self._logloss_digit.get(name, [None] * N_DIGITS)[d]
                if not buf or len(buf) < _MIN_HISTORY:
                    raw[name] = initial
                    continue
                loss = float(np.mean(buf))
                raw[name] = 1.0 / max(loss, 1e-3)

            total = sum(raw.values())
            if total <= 0:
                normalized = dict(self.initial_weights)
            else:
                normalized = {k: v / total for k, v in raw.items()}

            floored = {k: max(v, self.min_weight) for k, v in normalized.items()}
            total2 = sum(floored.values())
            for name in self.initial_weights:
                per_digit_weights[name][d] = floored[name] / total2

        return per_digit_weights
