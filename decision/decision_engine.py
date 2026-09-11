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

ARCHITECTURES = ("global", "specialist", "hybrid")


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
    architecture: str | None = None  # which of ARCHITECTURES actually produced this decision


@dataclass
class ArchitectureDecision:
    """The outcome of running the trade-quality gates for ONE architecture's
    probability vector, against a shared set of live quotes. Used both for
    the champion's real decision and for shadow-evaluating the other two
    architectures against the exact same quotes and gates."""
    calibrated_over: float
    calibrated_under: float
    over_edge_result: Any
    under_edge_result: Any
    agreement: dict[str, float]
    decision_label: str
    side: str | None
    chosen_edge_result: Any
    quality_score: float | None
    reason: str


def evaluate_architecture_decision(
    *, ensemble_vec: np.ndarray, predictions_for_agreement: dict[str, np.ndarray],
    calibration_over: CalibrationTracker, calibration_under: CalibrationTracker,
    regime: str, quote_over, quote_under, sample_size: int, over_barrier: int, under_barrier: int,
    mp_cfg: dict, risk_ok: bool, risk_reason: str | None,
) -> ArchitectureDecision:
    """Pure function: given one architecture's probability vector and a
    shared set of live quotes, runs the exact same mispricing/quality/
    abstention gates real trading uses. Factored out so
    learning/architecture_competition.py can apply IDENTICAL trade logic to
    all three competing architectures for a fair comparison -- the only
    thing that differs between architectures is the vector and its own
    calibration/agreement, never the gates themselves.
    """
    agreement = model_agreement(predictions_for_agreement, over_barrier, under_barrier)

    raw_over = float(np.sum(ensemble_vec[over_barrier + 1:]))
    raw_under = float(np.sum(ensemble_vec[:under_barrier]))
    calibrated_over = calibration_over.calibrate(raw_over)
    calibrated_under = calibration_under.calibrate(raw_under)

    over_edge_result = compute_edge(quote_over, calibrated_over) if quote_over else None
    under_edge_result = compute_edge(quote_under, calibrated_under) if quote_under else None

    candidates: list[tuple[str, Any, float, dict]] = []
    for side, edge_result, cal_tracker, agreement_key in (
        ("OVER", over_edge_result, calibration_over, "over_agreement"),
        ("UNDER", under_edge_result, calibration_under, "under_agreement"),
    ):
        if edge_result is None:
            continue
        mp_check = check_mispricing(
            edge_result,
            sample_size=sample_size,
            calibration_score=cal_tracker.quality_score(),
            model_agreement=agreement[agreement_key],
            minimum_edge=mp_cfg.get("minimum_edge", 0.03),
            minimum_probability=mp_cfg.get("minimum_probability", 0.55),
            minimum_calibration_score=mp_cfg.get("minimum_calibration_score", 0.6),
            minimum_model_agreement=mp_cfg.get("minimum_model_agreement", 0.6),
            minimum_sample_size=mp_cfg.get("minimum_sample_size", 300),
        )
        quality = score_trade(
            edge_result,
            calibration_score=cal_tracker.quality_score(),
            model_agreement=agreement[agreement_key],
            regime=regime,
            sample_size=sample_size,
            target_sample_size=mp_cfg.get("minimum_sample_size", 300) * 3,
        )
        abstain_reasons = collect_abstention_reasons(
            regime=regime,
            allow_low_confidence_regimes=mp_cfg.get("allow_low_confidence_regimes", False),
            risk_ok=risk_ok,
            risk_reason=risk_reason,
            quote_available=True,
            quality_score=quality.score,
            minimum_quality_score=mp_cfg.get("minimum_quality_score", 65),
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
        barrier = over_barrier if side == "OVER" else under_barrier
        decision_label = f"TRADE_{side}_{barrier}"
        reason = f"calibrated probability exceeds break-even with sufficient agreement and quality ({quality_score:.1f}/100)"
        chosen_edge_result = edge_result
    else:
        side = None
        decision_label = "NO_TRADE"
        chosen_edge_result = None
        all_reasons = []
        for _, _, _, meta in candidates:
            all_reasons.extend(meta["mispricing_reasons_failed"])
            all_reasons.extend(meta["abstain_reasons"])
        reason = ",".join(sorted(set(all_reasons))) if all_reasons else "no_positive_edge"
        quality_score = max((c[2] for c in candidates), default=None)

    return ArchitectureDecision(
        calibrated_over=calibrated_over, calibrated_under=calibrated_under,
        over_edge_result=over_edge_result, under_edge_result=under_edge_result,
        agreement=agreement, decision_label=decision_label, side=side,
        chosen_edge_result=chosen_edge_result, quality_score=quality_score, reason=reason,
    )


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

    def predict(self, state: SymbolState, bundle=None) -> tuple[dict[str, np.ndarray], Any]:
        if bundle is None:
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

    def combined_vector(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """The Global architecture's actual output: its 8 shared models
        combined via the promoted (champion) per-digit weights."""
        return combine(predictions, self.champion_weights)

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

    async def evaluate(self, client: DerivClient, state: SymbolState, competition: "ArchitectureCompetitionManager",
                        stake: float, currency: str, risk_ok: bool, risk_reason: str | None) -> Decision:
        now = time.time()
        symbol = state.symbol

        # Predict (and stash for next-tick learning) UNCONDITIONALLY, before
        # the trading sample-size gate below -- min_samples_per_symbol gates
        # whether Astra is willing to RISK MONEY, not whether it learns.
        # Gating this behind the trade-eligibility check would silently
        # regress the "learning begins immediately" property (see
        # tests/test_learning_starts_immediately.py) for every architecture
        # during the first ~300 ticks of a symbol's life.
        snapshot = competition.predict_all(state)
        competition.stash_pending(snapshot)

        if not state.has_min_samples(self.min_samples):
            return Decision(
                timestamp=now, symbol=symbol, probabilities={}, over_probability=0.0, under_probability=0.0,
                over_edge=None, under_edge=None, over_ev=None, under_ev=None, regime="UNKNOWN",
                model_agreement={}, calibration_quality={}, quality_score=None, decision="NO_TRADE",
                reason="insufficient_sample_size", sample_size=state.total_observed,
                architecture=competition.champion,
            )

        if not snapshot.global_predictions:
            return Decision(
                timestamp=now, symbol=symbol, probabilities={}, over_probability=0.0, under_probability=0.0,
                over_edge=None, under_edge=None, over_ev=None, under_ev=None, regime="UNKNOWN",
                model_agreement={}, calibration_quality={}, quality_score=None, decision="NO_TRADE",
                reason="no_ready_models", sample_size=state.total_observed,
                architecture=competition.champion,
            )

        champion_vec = snapshot.vectors[competition.champion]
        avg_model_std = (snapshot.raw_agreement["over_std"] + snapshot.raw_agreement["under_std"]) / 2.0
        regime_result = self.regime_detector.detect(snapshot.bundle, avg_model_std)
        regime = regime_result.regime

        quote_over = await get_quote(
            client, symbol, "DIGITOVER", self.over_barrier, stake, self.duration, self.duration_unit, currency,
        )
        quote_under = await get_quote(
            client, symbol, "DIGITUNDER", self.under_barrier, stake, self.duration, self.duration_unit, currency,
        )
        snapshot.quote_over = quote_over
        snapshot.quote_under = quote_under

        calibration = competition.calibration[competition.champion]
        champion_decision = evaluate_architecture_decision(
            ensemble_vec=champion_vec,
            predictions_for_agreement=snapshot.global_predictions,
            calibration_over=calibration["over"], calibration_under=calibration["under"],
            regime=regime, quote_over=quote_over, quote_under=quote_under,
            sample_size=state.total_observed, over_barrier=self.over_barrier, under_barrier=self.under_barrier,
            mp_cfg=self.mp_cfg, risk_ok=risk_ok, risk_reason=risk_reason,
        )

        # Shadow-evaluate the other two architectures against the SAME
        # quotes and gates, purely for comparison -- see
        # ArchitectureCompetitionManager for how these feed into promotion.
        for arch in ARCHITECTURES:
            if arch == competition.champion:
                snapshot.decisions[arch] = champion_decision
                continue
            cal = competition.calibration[arch]
            snapshot.decisions[arch] = evaluate_architecture_decision(
                ensemble_vec=snapshot.vectors[arch],
                predictions_for_agreement=snapshot.agreement_inputs[arch],
                calibration_over=cal["over"], calibration_under=cal["under"],
                regime=regime, quote_over=quote_over, quote_under=quote_under,
                sample_size=state.total_observed, over_barrier=self.over_barrier, under_barrier=self.under_barrier,
                mp_cfg=self.mp_cfg, risk_ok=risk_ok, risk_reason=risk_reason,
            )

        stake_out = stake if champion_decision.side is not None else None
        return Decision(
            timestamp=now,
            symbol=symbol,
            probabilities={str(i): float(champion_vec[i]) for i in range(10)},
            over_probability=champion_decision.calibrated_over,
            under_probability=champion_decision.calibrated_under,
            over_edge=champion_decision.over_edge_result.edge if champion_decision.over_edge_result else None,
            under_edge=champion_decision.under_edge_result.edge if champion_decision.under_edge_result else None,
            over_ev=champion_decision.over_edge_result.expected_value if champion_decision.over_edge_result else None,
            under_ev=champion_decision.under_edge_result.expected_value if champion_decision.under_edge_result else None,
            regime=regime,
            model_agreement=champion_decision.agreement,
            calibration_quality={
                "over": calibration["over"].quality_score(),
                "under": calibration["under"].quality_score(),
            },
            quality_score=champion_decision.quality_score,
            decision=champion_decision.decision_label,
            reason=f"[{competition.champion}] {champion_decision.reason}",
            sample_size=state.total_observed,
            stake=stake_out,
            quote_over=quote_over,
            quote_under=quote_under,
            raw_model_predictions={k: v.tolist() for k, v in snapshot.global_predictions.items()},
            architecture=competition.champion,
        )
