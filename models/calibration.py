from __future__ import annotations

import math
from collections import deque

import numpy as np


class CalibrationTracker:
    """Tracks (raw_predicted_probability, realized_outcome) pairs for one
    symbol+contract-side and fits an isotonic or Platt calibrator on a
    rolling buffer, refit periodically rather than on every observation.

    Also exposes Brier score / log loss / a simple reliability check so the
    decision engine and regime detector can down-weight or block trades when
    calibration quality is poor -- per spec section 13, an uncalibrated
    confidence score is never treated as a true probability.

    CRITICAL for any caller (digit or Rise/Fall): `outcome` must always be
    "did the thing raw_prob was a probability OF actually happen" (e.g. for
    Rise/Fall, "did price actually go up" -- 1/0), never "was the trade/
    prediction correct". Those two only coincide by accident when raw_prob
    is always computed for the same fixed side. Feeding "was correct" here
    silently biases the fitted calibrator toward whichever direction
    happened to win more often historically, rather than calibrating the
    probability itself -- confirmed as a real bug class in the source
    material this temperature-scaling addition was ported from (see
    fit_temperature()'s docstring below for the full explanation).
    """

    def __init__(self, method: str = "isotonic", buffer_size: int = 2000,
                 min_samples: int = 200, refit_every: int = 100,
                 use_temperature_blend: bool = False):
        self.method = method
        self.min_samples = min_samples
        self.refit_every = refit_every
        # Off by default -- does not change any existing (digit) caller's
        # behavior unless explicitly opted into. Ported from the Rise/Fall
        # source bot's ConfidenceCalibrator: a single-parameter temperature
        # fit is lower-variance than isotonic's flexible binned mapping,
        # especially with sparser data, so blending the two (70% isotonic /
        # 30% temperature, matching the source's own blend ratio) stabilizes
        # isotonic against overfitting local bins rather than replacing it.
        self.use_temperature_blend = use_temperature_blend
        self._raw: deque[float] = deque(maxlen=buffer_size)
        self._outcome: deque[int] = deque(maxlen=buffer_size)
        self._calibrator = None
        self._temperature: float = 1.0
        self._since_refit = 0
        self._brier_history: deque[float] = deque(maxlen=buffer_size)
        self._logloss_history: deque[float] = deque(maxlen=buffer_size)
        self._quality_cache: float | None = None
        self._quality_cache_dirty: bool = True

    def record(self, raw_prob: float, outcome: int) -> None:
        self._raw.append(raw_prob)
        self._outcome.append(outcome)
        self._brier_history.append((raw_prob - outcome) ** 2)
        p = min(max(raw_prob, 1e-6), 1 - 1e-6)
        self._logloss_history.append(float(-(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))))
        self._since_refit += 1
        self._quality_cache_dirty = True
        if len(self._raw) >= self.min_samples and self._since_refit >= self.refit_every:
            self._refit()
            self._since_refit = 0

    def _refit(self) -> None:
        if self.method != "none":
            X = np.array(self._raw)
            y = np.array(self._outcome)
            try:
                if self.method == "isotonic":
                    from sklearn.isotonic import IsotonicRegression
                    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    cal.fit(X, y)
                else:  # platt / logistic
                    from sklearn.linear_model import LogisticRegression
                    cal = LogisticRegression()
                    cal.fit(X.reshape(-1, 1), y)
                self._calibrator = cal
            except Exception:  # noqa: BLE001
                self._calibrator = None

        if self.use_temperature_blend:
            self._temperature = self._fit_temperature(np.array(self._raw), np.array(self._outcome))

    @staticmethod
    def _fit_temperature(confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """Fits T minimizing NLL(outcomes | sigmoid(logit(confidences)/T)).
        T > 1 softens (reduces overconfidence), T < 1 sharpens, T = 1 means
        no adjustment. Ported from the Rise/Fall source bot's
        ConfidenceCalibrator.fit_temperature() -- ported here as a private
        helper rather than a public API since Astra's calibrate() blends it
        with isotonic automatically when use_temperature_blend=True, callers
        never need to invoke this directly.

        `outcomes` here MUST be "did the outcome raw_prob predicted actually
        happen" (not "was the prediction/trade correct") -- see this class's
        docstring. This fit is symmetric in that mistake: it would happily
        converge to a T that makes the calibrator confidently biased toward
        whichever side historically won more, with no way to detect the
        mislabeling from the fit alone -- the correctness burden is entirely
        on the caller of record().
        """
        if len(confidences) < 50:
            return 1.0
        from scipy.optimize import minimize
        from scipy.special import expit as sigmoid

        p = np.clip(confidences, 0.01, 0.99)
        logits = np.log(p / (1 - p))
        y = outcomes.astype(float)

        def nll(t):
            t = max(t[0], 0.1)
            p_cal = sigmoid(logits / t)
            return -float(np.mean(y * np.log(p_cal + 1e-9) + (1 - y) * np.log(1 - p_cal + 1e-9)))

        try:
            res = minimize(nll, x0=[1.0], method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 200})
            return float(max(res.x[0], 0.1))
        except Exception:  # noqa: BLE001
            return 1.0

    def calibrate(self, raw_prob: float) -> float:
        primary = None
        if self._calibrator is not None:
            try:
                if self.method == "isotonic":
                    primary = float(self._calibrator.predict([raw_prob])[0])
                else:
                    primary = float(self._calibrator.predict_proba([[raw_prob]])[0][1])
            except Exception:  # noqa: BLE001
                primary = None

        if not self.use_temperature_blend:
            return primary if primary is not None else raw_prob

        p = min(max(raw_prob, 0.01), 0.99)
        temp_scaled = float(1.0 / (1.0 + math.exp(-math.log(p / (1 - p)) / self._temperature)))
        if primary is None:
            return temp_scaled
        # Matches the source material's own blend ratio: the flexible
        # binned calibrator (isotonic/platt) is trusted as primary, the
        # smoother single-parameter temperature fit contributes a
        # stabilizing 30% rather than overriding it.
        return float(np.clip(0.70 * primary + 0.30 * temp_scaled, 0.01, 0.99))

    def _calibrate_batch(self, raw: np.ndarray) -> np.ndarray:
        """Vectorized version of calibrate() for scoring a whole buffer at
        once (used by quality_score). This matters: calling calibrate() in a
        Python loop, once per buffered sample, on every tick, made
        quality_score() the single most expensive thing Astra did per tick
        once the calibration buffer grew into the hundreds -- one sklearn
        predict call per element instead of one call for the whole array.
        Profiling a 3000-tick backtest showed isotonic's per-element
        `_transform` alone eating ~70% of total runtime before this fix."""
        if self._calibrator is None:
            return raw
        try:
            if self.method == "isotonic":
                return np.asarray(self._calibrator.predict(raw), dtype=float)
            else:
                return np.asarray(self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
        except Exception:  # noqa: BLE001
            return raw

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None

    @property
    def sample_size(self) -> int:
        return len(self._raw)

    def quality_score(self) -> float:
        """0..1 calibration RELIABILITY score via (a simple binned
        approximation of) Expected Calibration Error -- NOT skill over a
        naive baseline. This distinction matters: a model whose stated
        probability is exactly the true base rate is, by definition,
        perfectly calibrated and IS good enough to trade on if the
        exchange's implied probability disagrees with it, even though it has
        "no skill" over predicting the base rate (there's no such thing as
        skill beyond the base rate for an iid source). An earlier version of
        this method used a Brier-skill-score-over-climatology formula, which
        scored that exact case at ~0 and blocked every trade on a real,
        tradeable, correctly-calibrated edge. Reliability (does a stated
        probability of p actually resolve to p, empirically) is the thing
        that matters for whether a probability can be trusted for wagering,
        independent of whether it required "extra skill" to produce."""
        if len(self._brier_history) < 30:
            return 0.3  # unproven; the decision engine's min_sample_size gate handles the rest

        if not self._quality_cache_dirty and self._quality_cache is not None:
            return self._quality_cache

        raw = np.array(self._raw)
        outcome = np.array(self._outcome)
        calibrated = self._calibrate_batch(raw)

        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(calibrated, bin_edges) - 1, 0, n_bins - 1)
        total = len(calibrated)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_idx == b
            count = int(mask.sum())
            if count < 5:
                continue
            mean_pred = float(calibrated[mask].mean())
            mean_actual = float(outcome[mask].mean())
            ece += (count / total) * abs(mean_pred - mean_actual)

        # a well-calibrated model typically has ECE well under 0.05; 0.3+ is
        # badly miscalibrated. Scale accordingly.
        score = 1.0 - ece / 0.3
        score = max(0.0, min(score, 1.0))
        self._quality_cache = score
        self._quality_cache_dirty = False
        return score

    def rolling_log_loss(self) -> float | None:
        if not self._logloss_history:
            return None
        return float(np.mean(self._logloss_history))
