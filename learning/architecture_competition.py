"""
Runs three competing digit-probability architectures side by side per
symbol, rather than betting on one philosophy upfront:

  A. "global"     -- shared multiclass models + adaptive per-digit weighting
                      (decision/decision_engine.py::SymbolPipeline)
  B. "specialist" -- 10 independent per-digit binary specialists
                      (models/digit_specialist.py)
  C. "hybrid"      -- an adaptively-weighted blend of A's and B's output

Only ONE architecture -- the current champion -- actually drives real
trades at any given time. The other two run in SHADOW mode: every tick,
their probability vectors are scored against the exact same live quotes and
the exact same trade-quality gates a real decision would use (see
decision/decision_engine.py::evaluate_architecture_decision), so the
comparison is apples-to-apples and never risks money on the non-champion
architectures. This mirrors the shadow-trading approach already used
elsewhere in this account (the adaptive touch/no-touch bot).

Every tick, all three architectures are scored on the seven dimensions
requested: log loss, Brier score, calibration (reliability), probability
stability (tick-to-tick vector movement), internal model agreement,
economic EV (unfiltered, "how good is the stated edge on average"), and
realized/hypothetical trading performance (P&L from only the trades that
would have passed the gates). Every `evaluate_every_n_ticks` observations,
the three are compared on a composite score (config-weighted, direction-
corrected, min-max normalized across the three) and the champion is
replaced ONLY if a challenger wins with a real margin AND the win holds up
across both halves of the evaluation window -- the same adversarial
stability discipline as learning/champion_challenger.py, applied one level
up, across architectures instead of across weight vectors.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import AstraConfig
from app.logging_setup import get_logger
from decision.decision_engine import ARCHITECTURES, ArchitectureDecision, SymbolPipeline, evaluate_architecture_decision
from features.feature_engine import build_features
from learning.champion_challenger import ChampionChallengerManager
from learning.online import PerformanceTracker
from models.calibration import CalibrationTracker
from models.digit_specialist import DigitSpecialistArchitecture
from models.ensemble import combine, model_agreement
from research.experiment_log import ExperimentLog
from state.rolling_state import SymbolState

logger = get_logger("learning.architecture_competition")

N_DIGITS = 10

# metric_name -> True if LOWER is better (raw value, before normalization)
_METRIC_DIRECTIONS = {
    "log_loss": True,
    "brier": True,
    "stability": True,
    "agreement": False,
    "economic_ev": False,
    "realized_pnl": False,
    "calibration": False,
}


@dataclass
class PredictionSnapshot:
    """Everything computed for one tick, across all three architectures.
    Stashed as `_pending` so the next tick's realized digit can be scored
    against it -- exactly the same pending/observe pattern used elsewhere
    (see SymbolPipeline / app/main.py), just carrying three architectures'
    worth of state instead of one."""
    bundle: Any
    global_predictions: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray]                 # architecture -> 10-digit vector
    agreement_inputs: dict[str, dict[str, np.ndarray]]  # architecture -> its own internal "models" for agreement scoring
    raw_agreement: dict[str, float]                 # Global's 8-model agreement, used for regime detection
    quote_over: Any = None
    quote_under: Any = None
    decisions: dict[str, ArchitectureDecision] = field(default_factory=dict)


class ArchitectureMetrics:
    """Rolling metric history for ONE architecture on ONE symbol."""

    def __init__(self, window: int = 1000):
        self.log_loss: deque[float] = deque(maxlen=window)
        self.brier: deque[float] = deque(maxlen=window)
        self.stability: deque[float] = deque(maxlen=window)
        self.agreement: deque[float] = deque(maxlen=window)
        self.economic_ev: deque[float] = deque(maxlen=window)
        self.realized_pnl: deque[float] = deque(maxlen=window)
        self._prev_vector: np.ndarray | None = None

    def record_prediction(self, vector: np.ndarray, actual_digit: int, agreement_value: float) -> None:
        # NOTE: every value appended here is wrapped in float()/explicitly
        # native -- `-np.log(...)` and friends return numpy.float64, and an
        # earlier version of a sibling tracker (learning/champion_challenger.py)
        # let numpy scalars leak into a dict that got sent straight to
        # Supabase, which silently failed on numpy.bool (doesn't subclass
        # Python's bool) for the entire time it ran. See
        # database/repository.py's _json_safe for the defensive fix and
        # champion_challenger.py for the source-level one -- native types
        # from the start here avoids needing to rely on either.
        p = float(np.clip(vector[actual_digit], 1e-9, 1.0))
        self.log_loss.append(float(-np.log(p)))
        onehot = np.zeros(N_DIGITS)
        onehot[actual_digit] = 1.0
        self.brier.append(float(np.sum((vector - onehot) ** 2)))
        if self._prev_vector is not None:
            self.stability.append(float(np.linalg.norm(vector - self._prev_vector)))
        self._prev_vector = vector.copy()
        self.agreement.append(float(agreement_value))

    def record_economic(self, ev: float | None) -> None:
        if ev is not None:
            self.economic_ev.append(float(ev))

    def record_realized(self, pnl: float | None) -> None:
        if pnl is not None:
            self.realized_pnl.append(float(pnl))

    def summary(self) -> dict[str, float | int | None]:
        def m(d: deque) -> float | None:
            return float(np.mean(d)) if d else None
        return {
            "log_loss": m(self.log_loss),
            "brier": m(self.brier),
            "stability": m(self.stability),
            "agreement": m(self.agreement),
            "economic_ev": m(self.economic_ev),
            "realized_pnl": float(np.sum(self.realized_pnl)) if self.realized_pnl else None,
            "n": len(self.log_loss),
        }


class ArchitectureCompetitionManager:
    def __init__(self, symbol: str, cfg: AstraConfig, repository=None):
        self.symbol = symbol
        self.cfg = cfg
        self.repository = repository

        self.global_pipeline = SymbolPipeline(symbol, cfg)
        self.specialist = DigitSpecialistArchitecture()
        # The hybrid blend reuses PerformanceTracker (per-digit inverse-loss
        # weighting) between exactly two "models": Global's output and
        # Specialist's output -- literally "shared models + digit
        # specialists" combined, per the requested Architecture C.
        self.hybrid_blend = PerformanceTracker(
            initial_weights={"global": 0.5, "specialist": 0.5}, min_weight=0.1, window=500,
        )

        cal_cfg = cfg.get("calibration", default={})
        self.calibration: dict[str, dict[str, CalibrationTracker]] = {
            arch: {
                "over": CalibrationTracker(method=cal_cfg.get("method", "isotonic"),
                                            min_samples=cal_cfg.get("min_samples_to_calibrate", 200),
                                            refit_every=cal_cfg.get("refit_every", 100)),
                "under": CalibrationTracker(method=cal_cfg.get("method", "isotonic"),
                                             min_samples=cal_cfg.get("min_samples_to_calibrate", 200),
                                             refit_every=cal_cfg.get("refit_every", 100)),
            }
            for arch in ARCHITECTURES
        }
        self.metrics: dict[str, ArchitectureMetrics] = {arch: ArchitectureMetrics() for arch in ARCHITECTURES}

        ac_cfg = cfg.get("architecture_competition", default={})
        self.evaluate_every_n = ac_cfg.get("evaluate_every_n_ticks", 200)
        self.min_samples_for_evaluation = ac_cfg.get("min_samples_for_evaluation", 200)
        self.min_promotion_margin = ac_cfg.get("min_promotion_margin", 0.03)
        self.composite_weights = ac_cfg.get("composite_weights", {
            "log_loss": 0.2, "brier": 0.15, "calibration": 0.2, "stability": 0.05,
            "agreement": 0.1, "economic_ev": 0.15, "realized_pnl": 0.15,
        })
        self._ticks_since_evaluation = 0

        self.champion: str = "global"
        if repository is not None:
            saved = repository.load_architecture_state(symbol)
            if saved and saved.get("champion_architecture") in ARCHITECTURES:
                self.champion = saved["champion_architecture"]
                logger.info("Restored champion architecture from Supabase", extra={"extra_fields": {
                    "symbol": symbol, "champion": self.champion,
                }})

            saved_specialists = repository.load_digit_specialist_state(symbol)
            if saved_specialists:
                try:
                    self.specialist.load_state(saved_specialists)
                    logger.info("Restored digit specialist state from Supabase", extra={"extra_fields": {
                        "symbol": symbol,
                    }})
                except Exception as exc:  # noqa: BLE001
                    # Never let a corrupt/incompatible snapshot (e.g. from an
                    # older schema version) block startup -- worst case,
                    # Architecture B relearns from its prior, same as if
                    # this table were empty.
                    logger.warning("Failed to restore digit specialist state, starting fresh",
                                    extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})

        # Still keep the existing within-Global weight-vector champion/
        # challenger running -- architecture-level competition (this class)
        # and weight-level competition (inside "global") are complementary,
        # not redundant: this class decides WHICH architecture drives
        # trades; champion_challenger.py keeps tuning Global's own weights
        # regardless of whether Global is currently in the lead.
        self.experiment_log = ExperimentLog(repository=repository)
        self.champion_challenger = ChampionChallengerManager(
            evaluate_every_n_trades=cfg.get("champion_challenger", "evaluate_every_n_trades", default=40),
            min_trades_to_evaluate=cfg.get("champion_challenger", "min_trades_to_evaluate", default=40),
            min_improvement=cfg.get("champion_challenger", "min_improvement", default=0.01),
            experiment_log=self.experiment_log,
        )

        self.over_barrier = cfg.get("contracts", "over_barrier", default=2)
        self.under_barrier = cfg.get("contracts", "under_barrier", default=7)

        self._pending: PredictionSnapshot | None = None

    def save_specialist_state(self) -> None:
        """Called periodically (see app/main.py's STATE_SNAPSHOT_EVERY_N_TICKS
        cadence) to persist Architecture B's learned state -- the same
        restart-survival guarantee astra_symbol_state already gives Global's
        champion_weights."""
        if self.repository is not None:
            self.repository.save_digit_specialist_state(self.symbol, self.specialist.get_state())

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_all(self, state: SymbolState) -> PredictionSnapshot:
        bundle = build_features(state, windows=self.cfg.get(
            "feature_windows", default=[20, 50, 100, 250, 500, 1000],
        ))

        global_predictions, _ = self.global_pipeline.predict(state, bundle)
        global_vec = (self.global_pipeline.combined_vector(global_predictions)
                      if global_predictions else np.full(N_DIGITS, 1.0 / N_DIGITS))

        bayes_vec, logit_vec, specialist_vec = self.specialist.predict_components(bundle)

        hybrid_weights = self.hybrid_blend.current_weights()
        hybrid_vec = combine({"global": global_vec, "specialist": specialist_vec}, hybrid_weights)

        vectors = {"global": global_vec, "specialist": specialist_vec, "hybrid": hybrid_vec}
        agreement_inputs = {
            "global": global_predictions,
            "specialist": {"specialist_bayes": bayes_vec, "specialist_logit": logit_vec},
            "hybrid": {"global": global_vec, "specialist": specialist_vec},
        }
        raw_agreement = (model_agreement(global_predictions, self.over_barrier, self.under_barrier)
                          if global_predictions else
                          {"over_std": 0.0, "under_std": 0.0, "over_agreement": 1.0, "under_agreement": 1.0})

        return PredictionSnapshot(
            bundle=bundle, global_predictions=global_predictions, vectors=vectors,
            agreement_inputs=agreement_inputs, raw_agreement=raw_agreement,
        )

    def stash_pending(self, snapshot: PredictionSnapshot) -> None:
        self._pending = snapshot

    def has_pending(self) -> bool:
        return self._pending is not None

    # ------------------------------------------------------------------ #
    # Learning (called once the tick's actual digit is known)
    # ------------------------------------------------------------------ #
    def observe_pending(self, state: SymbolState, actual_digit: int) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return

        self.global_pipeline.observe(
            state, pending.bundle, pending.global_predictions, actual_digit,
            self.over_barrier, self.under_barrier,
        )
        self.champion_challenger.on_trade_settled(self.symbol, self.global_pipeline)

        self.specialist.observe(pending.bundle, actual_digit)

        self.hybrid_blend.record(
            {"global": pending.vectors["global"], "specialist": pending.vectors["specialist"]}, actual_digit,
        )

        outcome_over = 1 if actual_digit > self.over_barrier else 0
        outcome_under = 1 if actual_digit < self.under_barrier else 0

        for arch in ARCHITECTURES:
            vec = pending.vectors[arch]
            agreement_dict = model_agreement(pending.agreement_inputs[arch], self.over_barrier, self.under_barrier)
            agreement_value = (agreement_dict["over_agreement"] + agreement_dict["under_agreement"]) / 2.0
            self.metrics[arch].record_prediction(vec, actual_digit, agreement_value)

            decision = pending.decisions.get(arch)
            if decision is None:
                continue

            self.calibration[arch]["over"].record(decision.calibrated_over, outcome_over)
            self.calibration[arch]["under"].record(decision.calibrated_under, outcome_under)

            best_ev = None
            for er in (decision.over_edge_result, decision.under_edge_result):
                if er is not None and (best_ev is None or er.expected_value > best_ev):
                    best_ev = er.expected_value
            self.metrics[arch].record_economic(best_ev)

            # Realized/hypothetical trading performance: only counts if this
            # architecture's OWN gated decision chose a side. For the
            # champion this uses the same quote its real trade was decided
            # against; for shadow architectures it's the hypothetical P&L
            # had they been live, against that same quote and the real
            # outcome -- never a real order for anything but the champion.
            if decision.side is not None and decision.chosen_edge_result is not None:
                won = (actual_digit > self.over_barrier) if decision.side == "OVER" else (actual_digit < self.under_barrier)
                quote = decision.chosen_edge_result.quote
                pnl = (quote.payout - quote.stake) if won else -quote.stake
                self.metrics[arch].record_realized(pnl)

        self._ticks_since_evaluation += 1
        if self._ticks_since_evaluation >= self.evaluate_every_n:
            self._ticks_since_evaluation = 0
            self._maybe_promote_architecture()

    # ------------------------------------------------------------------ #
    # Composite scoring & promotion
    # ------------------------------------------------------------------ #
    def _raw_metrics(self) -> dict[str, dict[str, float]]:
        raw = {}
        for arch in ARCHITECTURES:
            s = dict(self.metrics[arch].summary())
            cal = self.calibration[arch]
            s["calibration"] = (cal["over"].quality_score() + cal["under"].quality_score()) / 2.0
            raw[arch] = s
        return raw

    def _composite_scores(self, raw: dict[str, dict[str, float]]) -> dict[str, float]:
        normalized: dict[str, dict[str, float]] = {arch: {} for arch in ARCHITECTURES}
        for metric, lower_better in _METRIC_DIRECTIONS.items():
            values = {arch: (raw[arch].get(metric) if raw[arch].get(metric) is not None else 0.0)
                      for arch in ARCHITECTURES}
            lo, hi = min(values.values()), max(values.values())
            for arch in ARCHITECTURES:
                if hi - lo < 1e-12:
                    normalized[arch][metric] = 0.5  # all three tied on this metric
                    continue
                score01 = (values[arch] - lo) / (hi - lo)
                normalized[arch][metric] = (1.0 - score01) if lower_better else score01

        return {
            arch: float(sum(self.composite_weights.get(m, 0.0) * normalized[arch][m] for m in _METRIC_DIRECTIONS))
            for arch in ARCHITECTURES
        }

    def _stability_check(self, candidate: str) -> bool:
        """Same discipline as learning/champion_challenger.py: a challenger
        only gets promoted if it also beats the champion on log-loss (the
        least noisy of the seven metrics) in BOTH halves of the evaluation
        window, not just on average -- a win that only shows up in one half
        is more likely noise than a real, durable improvement."""
        champ_losses = list(self.metrics[self.champion].log_loss)
        cand_losses = list(self.metrics[candidate].log_loss)
        n = min(len(champ_losses), len(cand_losses))
        if n < self.min_samples_for_evaluation:
            return False
        champ_losses, cand_losses = champ_losses[-n:], cand_losses[-n:]
        mid = n // 2
        if mid == 0 or (n - mid) == 0:
            return False
        half1 = statistics.mean(champ_losses[:mid]) - statistics.mean(cand_losses[:mid])
        half2 = statistics.mean(champ_losses[mid:]) - statistics.mean(cand_losses[mid:])
        return bool(half1 > 0 and half2 > 0)

    def _maybe_promote_architecture(self) -> None:
        raw = self._raw_metrics()
        if any((raw[a].get("n") or 0) < self.min_samples_for_evaluation for a in ARCHITECTURES):
            return

        scores = self._composite_scores(raw)
        best = max(ARCHITECTURES, key=lambda a: scores[a])

        if best == self.champion:
            return

        margin = scores[best] - scores[self.champion]
        stable = self._stability_check(best)

        if stable and margin >= self.min_promotion_margin:
            self.experiment_log.record(
                symbol=self.symbol,
                hypothesis=f"architecture '{best}' outperforms current champion '{self.champion}'",
                metrics={"scores": scores, "raw_metrics": raw, "margin": margin, "stable": stable},
                decision="promote",
            )
            logger.info("Promoting architecture", extra={"extra_fields": {
                "symbol": self.symbol, "from": self.champion, "to": best, "scores": scores, "margin": margin,
            }})
            self.champion = best
            if self.repository is not None:
                self.repository.save_architecture_state(self.symbol, self.champion)
        else:
            self.experiment_log.record(
                symbol=self.symbol,
                hypothesis=f"architecture '{best}' vs current champion '{self.champion}'",
                metrics={"scores": scores, "raw_metrics": raw, "margin": margin, "stable": stable},
                decision="reject",
            )

        if self.repository is not None:
            for arch in ARCHITECTURES:
                self.repository.insert_architecture_performance(self.symbol, arch, raw[arch], scores[arch])
