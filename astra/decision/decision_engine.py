from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import AstraConfig
from app.logging_setup import get_logger
from decision.filters import collect_abstention_reasons
from decision.quality import score_trade
from features.feature_engine import build_features
from ingestion.deriv_client import DerivClient
from learning.online import PerformanceTracker
from models.calibration import CalibrationTracker
from models.ensemble import combine, model_agreement
from models.registry import SymbolModelRegistry
from pricing.edge import compute_edge
from pricing.mispricing import check_mispricing
from pricing.payout import get_quote
from regime.detector import RegimeDetector
from state.rolling_state import SymbolState

logger = get_logger("decision.decision_engine")


@dataclass
class Decision:
    timestamp: float
    symbol: str
    probabilities: dict[str, float]
    over_probability: float
    under_probability: float
    over_edge: float | None
    under_edge: float | None
    over_ev: float | None
    under_ev: float | None
    regime: str
    model_agreement: dict[str, float]
    calibration_quality: dict[str, float]
    quality_score: float | None
    decision: str  # TRADE_OVER_<n> | TRADE_UNDER_<n> | NO_TRADE
    reason: str
    sample_size: int
    stake: float | None = None
    quote_over: Any = None
    quote_under: Any = None
    raw_model_predictions: dict[str, list[float]] = field(default_factory=dict)


class SymbolPipeline:
    """Owns everything that carries per-symbol learned state: models,
    performance-based ensemble weights, and calibration for each contract
    side. One instance per traded symbol, created lazily on first tick."""

    def __init__(self, symbol: str, cfg: AstraConfig):
        self.symbol = symbol
        self.cfg = cfg
        self.registry = SymbolModelRegistry(
            symbol=symbol,
            max_markov_order=cfg.get("max_markov_order", default=3),
            rolling_window=500,
        )
        ens_cfg = cfg.get("ensemble", default={})
        self.performance = PerformanceTracker(
            initial_weights=ens_cfg.get("initial_weights", {}),
            min_weight=ens_cfg.get("min_weight", 0.02),
            window=ens_cfg.get("performance_window", 500),
        )
        cal_cfg = cfg.get("calibration", default={})
        self.calibration_over = CalibrationTracker(
            method=cal_cfg.get("method", "isotonic"),
            min_samples=cal_cfg.get("min_samples_to_calibrate", 200),
            refit_every=cal_cfg.get("refit_every", 100),
        )
        self.calibration_under = CalibrationTracker(
            method=cal_cfg.get("method", "isotonic"),
            min_samples=cal_cfg.get("min_samples_to_calibrate", 200),
            refit_every=cal_cfg.get("refit_every", 100),
        )
        # remember the last prediction so `observe()` (called once the digit
        # settles) can update models/calibration/performance consistently
        self._pending_predictions: dict[str, np.ndarray] | None = None
        self._pending_over_prob: float | None = None
        self._pending_under_prob: float | None = None
        # champion_weights: dict[model_name] -> length-10 array (one weight
        # per digit -- per-digit specialist weighting, see learning/online.py).
        # Starts as each model's scalar initial weight broadcast across all
        # 10 digits; the challenger (self.performance.current_weights(),
        # already per-digit) earns promotion into this via champion_challenger.py.
        self.champion_weights: dict[str, np.ndarray] = {
            name: np.full(10, w, dtype=float) for name, w in ens_cfg.get("initial_weights", {}).items()
        }
        from collections import deque
        self._ensemble_logloss_champion: deque[float] = deque(maxlen=2000)
        self._ensemble_logloss_challenger: deque[float] = deque(maxlen=2000)

    def predict(self, state: SymbolState) -> tuple[dict[str, np.ndarray], Any]:
        bundle = build_features(
            state,
            windows=self.cfg.get("feature_windows", default=[20, 50, 100, 250, 500, 1000]),
        )
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.registry.models.items():
            if not model.is_ready(state):
                continue
            try:
                predictions[name] = model.predict(state, bundle)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Model {name} predict failed", exc_info=exc,
                             extra={"extra_fields": {"symbol": state.symbol, "model": name}})
        return predictions, bundle

    def observe(self, state: SymbolState, bundle, predictions: dict[str, np.ndarray], actual_digit: int,
                over_barrier: int, under_barrier: int) -> None:
        for name, model in self.registry.models.items():
            try:
                model.observe(state, bundle, actual_digit)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Model {name} observe failed", exc_info=exc,
                             extra={"extra_fields": {"symbol": state.symbol, "model": name}})
        if predictions:
            self.performance.record(predictions, actual_digit)

            # champion/challenger bookkeeping: score BOTH the currently
            # promoted (champion) weight set and the latest performance-
            # based (challenger) weight set against this same realized
            # outcome, purely for comparison -- production predictions
            # always use the challenger weights via combine() in evaluate().
            champion_vec = combine(predictions, self.champion_weights)
            challenger_vec = combine(predictions, self.performance.current_weights())
            p_champ = float(np.clip(champion_vec[actual_digit], 1e-9, 1.0))
            p_chall = float(np.clip(challenger_vec[actual_digit], 1e-9, 1.0))
            # np.log() on a native float still returns numpy.float64 -- cast
            # back to a native float immediately so nothing downstream (in
            # particular learning/champion_challenger.py's stable/promote
            # booleans, computed via comparisons on values derived from
            # these deques) can inherit a numpy dtype. numpy.float64 happens
            # to serialize to JSON fine (it subclasses float), but a
            # numpy.bool_ produced downstream from a numpy-typed comparison
            # does not, and previously crashed every Supabase write to
            # astra_champion_challenger with "Object of type bool is not
            # JSON serializable".
            self._ensemble_logloss_champion.append(float(-np.log(p_champ)))
            self._ensemble_logloss_challenger.append(float(-np.log(p_chall)))

        if self._pending_over_prob is not None:
            outcome_over = 1 if actual_digit > over_barrier else 0
            self.calibration_over.record(self._pending_over_prob, outcome_over)
        if self._pending_under_prob is not None:
            outcome_under = 1 if actual_digit < under_barrier else 0
            self.calibration_under.record(self._pending_under_prob, outcome_under)


class DecisionEngine:
    def __init__(self, cfg: AstraConfig):
        self.cfg = cfg
        regime_cfg = cfg.get("regime", default={})
        self.regime_detector = RegimeDetector(
            entropy_high=regime_cfg.get("entropy_high", 0.985),
            entropy_low=regime_cfg.get("entropy_low", 0.90),
            chi_p_shift=regime_cfg.get("chi_square_p_shift", 0.01),
            model_agreement_unstable=regime_cfg.get("model_agreement_unstable", 0.12),
            min_window_for_regime=regime_cfg.get("min_window_for_regime", 200),
        )
        self.over_barrier = cfg.get("contracts", "over_barrier", default=2)
        self.under_barrier = cfg.get("contracts", "under_barrier", default=7)
        self.duration = cfg.get("contracts", "duration", default=1)
        self.duration_unit = cfg.get("contracts", "duration_unit", default="t")
        self.mp_cfg = cfg.get("mispricing", default={})
        self.min_samples = cfg.get("min_samples_per_symbol", default=300)

    async def evaluate(self, client: DerivClient, state: SymbolState, pipeline: SymbolPipeline,
                        stake: float, currency: str, risk_ok: bool, risk_reason: str | None) -> Decision:
        now = time.time()
        symbol = state.symbol

        if not state.has_min_samples(self.min_samples):
            return Decision(
                timestamp=now, symbol=symbol, probabilities={}, over_probability=0.0, under_probability=0.0,
                over_edge=None, under_edge=None, over_ev=None, under_ev=None, regime="UNKNOWN",
                model_agreement={}, calibration_quality={}, quality_score=None, decision="NO_TRADE",
                reason="insufficient_sample_size", sample_size=state.total_observed,
            )

        predictions, bundle = pipeline.predict(state)
        if not predictions:
            return Decision(
                timestamp=now, symbol=symbol, probabilities={}, over_probability=0.0, under_probability=0.0,
                over_edge=None, under_edge=None, over_ev=None, under_ev=None, regime="UNKNOWN",
                model_agreement={}, calibration_quality={}, quality_score=None, decision="NO_TRADE",
                reason="no_ready_models", sample_size=state.total_observed,
            )

        # Production always predicts with the promoted (champion) weights,
        # not the raw live performance-based weights -- those are the
        # challenger, and only replace the champion once
        # learning/champion_challenger.py has validated the improvement is
        # real and stable (see that module's promotion criteria).
        weights = pipeline.champion_weights
        ensemble_vec = combine(predictions, weights)
        agreement = model_agreement(predictions, self.over_barrier, self.under_barrier)

        raw_over = float(np.sum(ensemble_vec[self.over_barrier + 1:]))
        raw_under = float(np.sum(ensemble_vec[:self.under_barrier]))

        pipeline._pending_over_prob = raw_over
        pipeline._pending_under_prob = raw_under

        calibrated_over = pipeline.calibration_over.calibrate(raw_over)
        calibrated_under = pipeline.calibration_under.calibrate(raw_under)

        avg_model_std = (agreement["over_std"] + agreement["under_std"]) / 2.0
        regime_result = self.regime_detector.detect(bundle, avg_model_std)
        regime = regime_result.regime

        quote_over = await get_quote(
            client, symbol, "DIGITOVER", self.over_barrier, stake, self.duration, self.duration_unit, currency,
        )
        quote_under = await get_quote(
            client, symbol, "DIGITUNDER", self.under_barrier, stake, self.duration, self.duration_unit, currency,
        )

        over_edge_result = compute_edge(quote_over, calibrated_over) if quote_over else None
        under_edge_result = compute_edge(quote_under, calibrated_under) if quote_under else None

        candidates: list[tuple[str, Any, float, dict]] = []

        for side, edge_result, cal_tracker, agreement_key in (
            ("OVER", over_edge_result, pipeline.calibration_over, "over_agreement"),
            ("UNDER", under_edge_result, pipeline.calibration_under, "under_agreement"),
        ):
            if edge_result is None:
                continue
            mp_check = check_mispricing(
                edge_result,
                sample_size=state.total_observed,
                calibration_score=cal_tracker.quality_score(),
                model_agreement=agreement[agreement_key],
                minimum_edge=self.mp_cfg.get("minimum_edge", 0.03),
                minimum_probability=self.mp_cfg.get("minimum_probability", 0.55),
                minimum_calibration_score=self.mp_cfg.get("minimum_calibration_score", 0.6),
                minimum_model_agreement=self.mp_cfg.get("minimum_model_agreement", 0.6),
                minimum_sample_size=self.mp_cfg.get("minimum_sample_size", 300),
            )
            quality = score_trade(
                edge_result,
                calibration_score=cal_tracker.quality_score(),
                model_agreement=agreement[agreement_key],
                regime=regime,
                sample_size=state.total_observed,
                target_sample_size=self.mp_cfg.get("minimum_sample_size", 300) * 3,
            )
            abstain_reasons = collect_abstention_reasons(
                regime=regime,
                allow_low_confidence_regimes=self.mp_cfg.get("allow_low_confidence_regimes", False),
                risk_ok=risk_ok,
                risk_reason=risk_reason,
                quote_available=True,
                quality_score=quality.score,
                minimum_quality_score=self.mp_cfg.get("minimum_quality_score", 65),
            )
            eligible = mp_check.passes and not abstain_reasons
            candidates.append((side, edge_result, quality.score, {
                "mispricing_reasons_failed": mp_check.reasons_failed,
                "abstain_reasons": abstain_reasons,
                "eligible": eligible,
            }))

        chosen = None
        for side, edge_result, quality_score, meta in candidates:
            if not meta["eligible"]:
                continue
            if chosen is None or edge_result.expected_value > chosen[1].expected_value:
                chosen = (side, edge_result, quality_score, meta)

        if chosen is not None:
            side, edge_result, quality_score, meta = chosen
            barrier = self.over_barrier if side == "OVER" else self.under_barrier
            decision_label = f"TRADE_{side}_{barrier}"
            reason = f"calibrated probability exceeds break-even with sufficient agreement and quality ({quality_score:.1f}/100)"
        else:
            decision_label = "NO_TRADE"
            all_reasons = []
            for side, edge_result, quality_score, meta in candidates:
                all_reasons.extend(meta["mispricing_reasons_failed"])
                all_reasons.extend(meta["abstain_reasons"])
            reason = ",".join(sorted(set(all_reasons))) if all_reasons else "no_positive_edge"
            quality_score = max((c[2] for c in candidates), default=None)

        return Decision(
            timestamp=now,
            symbol=symbol,
            probabilities={str(i): float(ensemble_vec[i]) for i in range(10)},
            over_probability=calibrated_over,
            under_probability=calibrated_under,
            over_edge=over_edge_result.edge if over_edge_result else None,
            under_edge=under_edge_result.edge if under_edge_result else None,
            over_ev=over_edge_result.expected_value if over_edge_result else None,
            under_ev=under_edge_result.expected_value if under_edge_result else None,
            regime=regime,
            model_agreement=agreement,
            calibration_quality={
                "over": pipeline.calibration_over.quality_score(),
                "under": pipeline.calibration_under.quality_score(),
            },
            quality_score=quality_score,
            decision=decision_label,
            reason=reason,
            sample_size=state.total_observed,
            stake=stake if chosen is not None else None,
            quote_over=quote_over,
            quote_under=quote_under,
            raw_model_predictions={k: v.tolist() for k, v in predictions.items()},
        )
