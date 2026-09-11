"""
Drift detection for whether a fitted model's assumptions still hold against
live data. Ported from risefall_bot_v4_hmm_gbm.py's DriftDetector, with two
of its own hard-won production bugfixes kept intact (see the comments on
snapshot_reference() and check_all() below) -- both were found from actual
production behavior, not a design review, and are exactly the kind of thing
that looks safe to "simplify" away without that history. A third fix, found
DURING this port (not present in the source -- see the "PORT-TIME FIX"
comment on _psi() below), corrects a blind spot where PSI could report
near-zero drift on the MOST extreme possible shift (a live distribution
sitting entirely outside the reference's observed range).

Refactored to a self-contained instance (one per symbol, or per symbol+side)
holding its own state, rather than the source's static-methods-against-a-
shared-TradeState design -- matches Astra's existing per-symbol-tracker
pattern (see models/calibration.py's CalibrationTracker) and makes this
independently testable without a large shared state object.

Three independent checks:
  KS-test: distribution of recent live log-returns vs. a reference snapshot
           taken at last (re)calibration. A significant shift means the
           fitted GARCH/HMM/OU parameters are stale.
  PSI:     Population Stability Index on confidence scores -- whether the
           model's OUTPUT distribution has shifted vs. its calibration-time
           distribution, independent of whether the input distribution has.
  CUSUM:   cumulative sum of (0.5 - win_indicator), detects a sustained
           below-50% win-rate run faster than a rolling average would.
           NOTE on the default parameters (cusum_drift=0.03,
           cusum_threshold=4.0, both taken from the source as-is): measured
           by direct simulation, this fires on a genuinely FAIR (50/50) win
           rate roughly 84% of the time within 200 draws (average ~114
           draws to fire), vs. ~28 draws on a genuinely losing (35% win
           rate) run -- correctly never fires on an all-wins run, and fires
           markedly faster the worse the true win rate is, but it is NOT a
           rare-event alarm over a realistic trade-count horizon. In the
           source, the consequence of firing is mild (DRIFT_STAKE_REDUCTION
           halves stake, doesn't block trading), which likely made this
           level of sensitivity an acceptable "err toward caution" trade-off
           there. Reconsider cusum_drift/cusum_threshold before relying on
           update_cusum() for anything with a more severe consequence than
           a stake reduction (e.g. blocking trading outright).

No internal logging -- unlike the source, which printed directly on each
fire. Callers should log when check_all()/check_ks()/check_psi() return
True; keeping this module side-effect-free makes it testable without
mocking a logger.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import ks_2samp


class DriftDetector:
    def __init__(self, ks_p_threshold: float = 0.05, psi_threshold: float = 0.20,
                 cusum_drift: float = 0.03, cusum_threshold: float = 4.0,
                 consecutive_required: int = 3, confidence_history_size: int = 200):
        self.ks_p_threshold = ks_p_threshold
        self.psi_threshold = psi_threshold
        self.cusum_drift = cusum_drift
        self.cusum_threshold = cusum_threshold
        self.consecutive_required = consecutive_required

        self._reference_returns: np.ndarray | None = None
        self._reference_confidences: np.ndarray | None = None
        self._confidence_history: deque[float] = deque(maxlen=confidence_history_size)
        self._cusum_stat: float = 0.0
        self._consecutive_fires: int = 0
        self._degraded: bool = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def check_ks(self, live_returns: np.ndarray) -> bool:
        if self._reference_returns is None or len(self._reference_returns) < 50 or len(live_returns) < 50:
            return False
        live_r = live_returns[-200:] if len(live_returns) > 200 else live_returns
        _, pval = ks_2samp(self._reference_returns, live_r)
        return bool(pval < self.ks_p_threshold)

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        # PORT-TIME FIX (found while testing this port, not present in the
        # source, which used `bins[0] -= 1e-9; bins[-1] += 1e-9` -- a tiny
        # epsilon nudge, not a real fix): np.histogram silently DROPS any
        # value falling outside its bin range entirely, rather than counting
        # it in the nearest edge bin. With percentile-derived interior bins,
        # a live confidence distribution that has shifted so far it no
        # longer overlaps the reference's observed range AT ALL produces an
        # all-zero raw histogram -- which, after the uniform +1 smoothing
        # below, looks statistically indistinguishable from the reference's
        # own roughly-uniform per-bin counts. Confirmed numerically: an
        # actual distribution sitting entirely outside the reference's range
        # computed PSI=0.0 (no drift detected) under the source's epsilon
        # version, vs PSI=4.2 (correctly, decisively over threshold) once
        # the outer edges are extended to +/-infinity instead -- the
        # standard convention in PSI implementations specifically to avoid
        # this blind spot. This is exactly backwards from what a stability
        # index should do: the MORE extreme the shift, the LESS likely the
        # epsilon version was to detect it at all.
        bins[0] = -np.inf
        bins[-1] = np.inf
        exp_counts = np.histogram(expected, bins=bins)[0] + 1
        act_counts = np.histogram(actual, bins=bins)[0] + 1
        exp_pct = exp_counts / exp_counts.sum()
        act_pct = act_counts / act_counts.sum()
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def check_psi(self, live_confidence: float) -> bool:
        self._confidence_history.append(live_confidence)
        if len(self._confidence_history) < 100:
            return False
        if self._reference_confidences is None or len(self._reference_confidences) < 50:
            return False
        psi = self._psi(self._reference_confidences, np.array(self._confidence_history))
        return psi > self.psi_threshold

    def update_cusum(self, won: bool) -> bool:
        outcome = 1.0 if won else 0.0
        self._cusum_stat = max(0.0, self._cusum_stat + (0.5 - self.cusum_drift) - outcome)
        return self._cusum_stat > self.cusum_threshold

    def snapshot_reference(self, returns: np.ndarray, confidences) -> None:
        """Call after each (re)calibration to reset the reference
        distributions.

        BUGFIX (kept from source, found in production there): this used to
        leave the confidence-history deque untouched across a
        recalibration. check_psi() only needs len(history) >= 100 before it
        starts comparing -- with the deque never cleared, that minimum was
        already satisfied by up to `confidence_history_size` STALE readings
        carried over from before this calibration, including the very
        readings that caused the drift trigger in the first place. Result:
        the first live PSI check right after a fresh recalibration compared
        the new reference against old, already-known-shifted data -- a
        self-perpetuating lock (recalibrate -> instantly re-fail against
        stale history -> recalibrate again minutes later, trading unlocked
        only in the gap). Clearing the deque here forces check_psi() to
        accumulate genuinely NEW post-calibration readings before it can
        fire again.
        """
        self._reference_returns = np.asarray(returns[-500:]).copy()
        self._reference_confidences = np.array(list(confidences)[-200:])
        self._cusum_stat = 0.0
        self._degraded = False
        self._confidence_history.clear()
        # Reset the consecutive-fire streak too, for the same reason as
        # above -- otherwise a streak already near consecutive_required from
        # just before this calibration could re-trigger almost immediately.
        self._consecutive_fires = 0

    def check_all(self, live_returns: np.ndarray, live_confidence: float) -> bool:
        """Runs KS + PSI with a debounce requirement (see below); CUSUM is
        driven separately by update_cusum() on trade outcomes, not folded
        into the same streak counter here -- CUSUM already requires a
        sustained run below its own accumulator threshold to fire, so
        requiring persistence on top of persistence would be redundant.
        Returns True if drift is judged SUSTAINED (the detector is in a
        degraded state).

        BUGFIX (kept from source, found in production there): this used to
        latch `degraded=True` on the VERY FIRST fire of KS or PSI, with no
        decay, cleared only by the next full recalibration. Checked
        frequently across many symbols, each getting hundreds of
        independent check opportunities between calibrations, a one-shot
        latch makes it almost inevitable that every symbol accumulates at
        least one noisy blip regardless of whether anything is
        persistently wrong -- confirmed in the source's own production
        logs, where every recalibration after the first listed every
        symbol by name. Requiring `consecutive_required` fires in a row
        (any non-fire resets the streak to 0) means a single noisy read
        can't lock a symbol out until the next recalibration -- only a
        signal that keeps showing up, check after check, can.
        """
        ks_fired = self.check_ks(live_returns)
        psi_fired = self.check_psi(live_confidence)
        if ks_fired or psi_fired:
            self._consecutive_fires += 1
        else:
            self._consecutive_fires = 0

        if self._consecutive_fires >= self.consecutive_required and not self._degraded:
            self._degraded = True
        return self._degraded
