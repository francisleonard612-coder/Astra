"""
All Supabase reads/writes go through this module. Every call is wrapped so a
transient DB failure degrades the bot's persistence (no history for that
event) rather than crashing the trading loop -- Astra keeps trading and
learning in memory even if Supabase is briefly unreachable.

Every payload also goes through `_json_safe()` before being sent. This
exists because of a real bug caught from a live deployment log: numpy
scalar types (numpy.float64, and especially numpy.bool -- which, unlike
numpy.float64, does NOT subclass Python's built-in bool, since bool can't be
subclassed at all) silently killed every `insert_experiment` call for the
entire time champion/challenger was evaluating, because `_safe()` catches
and logs the exception rather than crashing -- so the trading loop kept
running with no visible symptom other than an empty astra_experiment_log
table. `_json_safe()` recursively converts numpy scalars/arrays to native
Python types so this class of bug can't silently recur from anywhere else
in the codebase (including the architecture-competition metrics below,
which do similar numpy-derived comparisons).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.logging_setup import get_logger
from decision.decision_engine import Decision
from execution.orders import TradeResult
from research.experiment_log import ExperimentRecord

logger = get_logger("database.repository")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):  # covers numpy.float64, numpy.bool, numpy.int64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


class Repository:
    def __init__(self, client, persist_ticks: bool = True):
        self.client = client
        self.persist_ticks = persist_ticks

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _safe(self, fn, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Supabase write failed", exc_info=exc, extra={"extra_fields": {"fn": getattr(fn, "__name__", "?")}})
            return None

    def upsert_symbol(self, symbol: str, market: str = "synthetic_index") -> None:
        row = _json_safe({"symbol": symbol, "market": market, "last_seen_at": "now()"})
        self._safe(lambda: self.client.table("astra_symbols").upsert(row).execute())

    def insert_tick(self, symbol: str, epoch: int, quote: float, digit: int) -> None:
        if not self.persist_ticks:
            return
        row = _json_safe({"symbol": symbol, "epoch": epoch, "quote": quote, "digit": digit})
        self._safe(lambda: self.client.table("astra_ticks").insert(row).execute())

    def insert_prediction(self, decision: Decision) -> int | None:
        row = _json_safe({
            "symbol": decision.symbol,
            "probabilities": decision.probabilities,
            "over_probability": decision.over_probability,
            "under_probability": decision.under_probability,
            "over_edge": decision.over_edge,
            "under_edge": decision.under_edge,
            "over_ev": decision.over_ev,
            "under_ev": decision.under_ev,
            "regime": decision.regime,
            "model_agreement": decision.model_agreement,
            "calibration_quality": decision.calibration_quality,
            "quality_score": decision.quality_score,
            "decision": decision.decision,
            "reason": decision.reason,
            "sample_size": decision.sample_size,
            "raw_model_predictions": decision.raw_model_predictions,
            "architecture": getattr(decision, "architecture", None),
        })
        result = self._safe(lambda: self.client.table("astra_predictions").insert(row).execute())
        if result and result.data:
            return result.data[0].get("id")
        return None

    def insert_trade(self, trade: TradeResult, prediction_id: int | None = None) -> None:
        row = _json_safe({
            "symbol": trade.symbol, "contract_type": trade.contract_type, "barrier": trade.barrier,
            "stake": trade.stake, "payout": trade.payout, "contract_id": trade.contract_id,
            "won": trade.won, "pnl": trade.pnl, "error": trade.error, "prediction_id": prediction_id,
        })
        self._safe(lambda: self.client.table("astra_trades").insert(row).execute())

    def insert_regime_event(self, symbol: str, regime: str, detail: dict) -> None:
        row = _json_safe({"symbol": symbol, "regime": regime, "detail": detail})
        self._safe(lambda: self.client.table("astra_regime_log").insert(row).execute())

    def insert_model_performance(self, symbol: str, model_name: str, rolling_log_loss: float | None,
                                  weight_per_digit) -> None:
        weights_arr = np.asarray(weight_per_digit, dtype=float)
        row = _json_safe({
            "symbol": symbol, "model_name": model_name, "rolling_log_loss": rolling_log_loss,
            "weight": float(weights_arr.mean()),
            "weight_per_digit": {str(i): float(weights_arr[i]) for i in range(len(weights_arr))},
        })
        self._safe(lambda: self.client.table("astra_model_performance").insert(row).execute())

    def insert_champion_challenger(self, symbol: str, champion_loss: float, challenger_loss: float,
                                    improvement: float, stable: bool, promoted: bool, weights: dict) -> None:
        row = _json_safe({
            "symbol": symbol, "champion_log_loss": champion_loss, "challenger_log_loss": challenger_loss,
            "improvement": improvement, "stable": stable, "promoted": promoted, "weights": weights,
        })
        self._safe(lambda: self.client.table("astra_champion_challenger").insert(row).execute())

    def insert_architecture_performance(self, symbol: str, architecture: str, metrics: dict,
                                         composite_score: float | None = None) -> None:
        row = _json_safe({
            "symbol": symbol, "architecture": architecture, "composite_score": composite_score, **metrics,
        })
        self._safe(lambda: self.client.table("astra_architecture_performance").insert(row).execute())

    def save_architecture_state(self, symbol: str, champion_architecture: str) -> None:
        row = _json_safe({
            "symbol": symbol, "champion_architecture": champion_architecture, "updated_at": "now()",
        })
        self._safe(lambda: self.client.table("astra_architecture_state").upsert(row).execute())

    def load_architecture_state(self, symbol: str) -> dict[str, Any] | None:
        result = self._safe(
            lambda: self.client.table("astra_architecture_state").select("*").eq("symbol", symbol).execute()
        )
        if result and result.data:
            return result.data[0]
        return None

    def save_digit_specialist_state(self, symbol: str, specialist_states: list[dict]) -> None:
        row = _json_safe({
            "symbol": symbol, "specialists": specialist_states, "updated_at": "now()",
        })
        self._safe(lambda: self.client.table("astra_digit_specialist_state").upsert(row).execute())

    def load_digit_specialist_state(self, symbol: str) -> list[dict] | None:
        result = self._safe(
            lambda: self.client.table("astra_digit_specialist_state").select("*").eq("symbol", symbol).execute()
        )
        if result and result.data:
            return result.data[0].get("specialists")
        return None

    def insert_experiment(self, rec: ExperimentRecord) -> None:
        row = _json_safe({
            "symbol": rec.symbol, "hypothesis": rec.hypothesis, "metrics": rec.metrics, "decision": rec.decision,
        })
        self._safe(lambda: self.client.table("astra_experiment_log").insert(row).execute())

    def insert_risk_event(self, symbol: str | None, event_type: str, detail: dict) -> None:
        row = _json_safe({"symbol": symbol, "event_type": event_type, "detail": detail})
        self._safe(lambda: self.client.table("astra_risk_events").insert(row).execute())

    def insert_system_event(self, component: str, event_type: str, detail: dict | None = None) -> None:
        row = _json_safe({"component": component, "event_type": event_type, "detail": detail or {}})
        self._safe(lambda: self.client.table("astra_system_events").insert(row).execute())

    def save_symbol_state(self, symbol: str, total_observed: int, recent_digits: list[int],
                           champion_weights: dict, challenger_weights: dict) -> None:
        row = _json_safe({
            "symbol": symbol, "total_observed": total_observed, "recent_digits": recent_digits,
            "champion_weights": champion_weights, "challenger_weights": challenger_weights,
            "updated_at": "now()",
        })
        self._safe(lambda: self.client.table("astra_symbol_state").upsert(row).execute())

    def load_symbol_state(self, symbol: str) -> dict[str, Any] | None:
        result = self._safe(lambda: self.client.table("astra_symbol_state").select("*").eq("symbol", symbol).execute())
        if result and result.data:
            return result.data[0]
        return None

    def prune_old_ticks(self, retention_hours: int) -> None:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - retention_hours * 3600))
        self._safe(lambda: self.client.table("astra_ticks").delete().lt("created_at", cutoff).execute())
