"""
Online stacking model: combines whatever upstream signals are available
into a single P(direction=+1) estimate, learned via online SGD (updated
after every resolved trade) with periodic batch retrain from a rolling
buffer to avoid cold-start lag right after a reset.

Ported the MECHANISM only, not the source's specific 16-feature vector
(risefall_bot_v4_hmm_gbm.py's MetaLearner.LAYER_KEYS: markov, hmm, hawkes,
ou, hurst, arfima, kalman, copula, rsi, srsi, adx, boll, zscore, te, jump,
post_jump). Most of those are the technical-indicator grab-bag already
flagged as not worth porting (weak theoretical grounding against an audited
CSPRNG-driven process), or signals Astra doesn't have fitted anywhere yet
(HMM regime, OU reversion, and Hawkes clustering are dormant hooks in
pricing/monte_carlo_duration.py, not fitted models). Hard-coding that
feature set here would mean training against mostly-zero inputs. Instead
this class is parameterized by n_features -- callers build whatever feature
vector Astra actually has (e.g. [mc_win_probability, calibration_quality,
drift_detector_degraded, ...]) themselves; this class only owns the
online/batch learning mechanism.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import expit as sigmoid


class OnlineMetaLearner:
    def __init__(self, n_features: int, min_samples: int = 50, learning_rate: float = 0.01,
                 l2: float = 0.001, buffer_size: int = 2000):
        self.n_features = n_features
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.l2 = l2
        self._buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=buffer_size)
        self._w: np.ndarray | None = None
        self._b: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self._w is not None and len(self._buffer) >= self.min_samples

    def predict(self, x) -> float | None:
        """Returns P(direction=+1), or None below min_samples -- signals the
        caller to fall back to whatever else it uses (e.g. the raw MC
        probability) until enough examples exist."""
        if not self.is_ready:
            return None
        return float(sigmoid(np.dot(self._w, np.asarray(x, dtype=float)) + self._b))

    def update(self, x, y: float) -> None:
        """`y` must be 1.0 if direction actually went +1 (e.g. price rose),
        0.0 otherwise -- the REALIZED outcome, never "was the trade/bet
        correct" (same correctness requirement, and same reasoning, as
        CalibrationTracker.record() -- see its docstring in
        models/calibration.py for the full explanation of why those two
        differ and why conflating them silently biases the fit)."""
        x = np.asarray(x, dtype=float)
        self._buffer.append((x.copy(), y))
        if self._w is None:
            self._w = np.zeros(self.n_features)
        if len(self._buffer) < self.min_samples:
            return
        p = float(sigmoid(np.dot(self._w, x) + self._b))
        err = p - y
        grad_w = err * x + self.l2 * self._w
        grad_b = err
        self._w = self._w - self.learning_rate * grad_w
        self._b = self._b - self.learning_rate * grad_b

    def retrain_from_buffer(self, epochs: int = 50, lr_scale: float = 0.3) -> None:
        """Full batch retrain over the rolling buffer -- call after a
        recalibration (e.g. alongside DriftDetector.snapshot_reference())
        to avoid cold-start lag rather than relying purely on the
        incremental per-example update() to catch back up."""
        if len(self._buffer) < self.min_samples:
            return
        X = np.array([x for x, _ in self._buffer])
        y = np.array([lbl for _, lbl in self._buffer])
        w = self._w if self._w is not None else np.zeros(self.n_features)
        b = self._b
        lr = self.learning_rate * lr_scale
        for _ in range(epochs):
            preds = sigmoid(X @ w + b)
            errs = preds - y
            w = w - lr * (X.T @ errs / len(y) + self.l2 * w)
            b = b - lr * float(np.mean(errs))
        self._w = w
        self._b = float(b)
