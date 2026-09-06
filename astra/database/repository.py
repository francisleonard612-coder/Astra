"""
All Supabase reads/writes go through this module. Every call is wrapped so a
transient DB failure degrades the bot's persistence (no history for that
event) rather than crashing the trading loop -- Astra keeps trading and
learning in memory even if Supabase is briefly unreachable.
"""
from __future__ import annotations

import time
from typing import Any

from app.logging_setup import get_logger
from decision.decision_engine import Decision
from execution.orders import TradeResult
from research.experiment_log import ExperimentRecord

logger = get_logger("database.repository")


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
        self._safe(lambda: self.client.table("astra_symbols").upsert({
            "symbol": symbol, "market": market, "last_seen_at": "now()",
        }).execute())

    def insert_tick(self, symbol: str, epoch: int, quote: float, digit: int) -> None:
        if not self.persist_ticks:
            return
        self._safe(lambda: self.client.table("astra_ticks").insert({
            "symbol": symbol, "epoch": epoch, "quote": quote, "digit": digit,
        }).execute())

    def insert_prediction(self, decision: Decision) -> int | None:
        row = {
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
        }
        result = self._safe(lambda: self.client.table("astra_predictions").insert(row).execute())
        if result and result.data:
            return result.data[0].get("id")
        return None

    def insert_trade(self, trade: TradeResult, prediction_id: int | None = None) -> None:
        self._safe(lambda: self.client.table("astra_trades").insert({
            "symbol": trade.symbol, "contract_type": trade.contract_type, "barrier": trade.barrier,
            "stake": trade.stake, "payout": trade.payout, "contract_id": trade.contract_id,
            "won": trade.won, "pnl": trade.pnl, "error": trade.error, "prediction_id": prediction_id,
        }).execute())

    def insert_regime_event(self, symbol: str, regime: str, detail: dict) -> None:
        self._safe(lambda: self.client.table("astra_regime_log").insert({
            "symbol": symbol, "regime": regime, "detail": detail,
        }).execute())

    def insert_model_performance(self, symbol: str, model_name: str, rolling_log_loss: float | None,
                                  weight_per_digit) -> None:
        import numpy as np
        weights_arr = np.asarray(weight_per_digit, dtype=float)
        self._safe(lambda: self.client.table("astra_model_performance").insert({
            "symbol": symbol, "model_name": model_name, "rolling_log_loss": rolling_log_loss,
            "weight": float(weights_arr.mean()),
            "weight_per_digit": {str(i): float(weights_arr[i]) for i in range(len(weights_arr))},
        }).execute())

    def insert_champion_challenger(self, symbol: str, champion_loss: float, challenger_loss: float,
                                    improvement: float, stable: bool, promoted: bool, weights: dict) -> None:
        self._safe(lambda: self.client.table("astra_champion_challenger").insert({
            "symbol": symbol, "champion_log_loss": champion_loss, "challenger_log_loss": challenger_loss,
            "improvement": improvement, "stable": stable, "promoted": promoted, "weights": weights,
        }).execute())

    def insert_experiment(self, rec: ExperimentRecord) -> None:
        self._safe(lambda: self.client.table("astra_experiment_log").insert({
            "symbol": rec.symbol, "hypothesis": rec.hypothesis, "metrics": rec.metrics, "decision": rec.decision,
        }).execute())

    def insert_risk_event(self, symbol: str | None, event_type: str, detail: dict) -> None:
        self._safe(lambda: self.client.table("astra_risk_events").insert({
            "symbol": symbol, "event_type": event_type, "detail": detail,
        }).execute())

    def insert_system_event(self, component: str, event_type: str, detail: dict | None = None) -> None:
        self._safe(lambda: self.client.table("astra_system_events").insert({
            "component": component, "event_type": event_type, "detail": detail or {},
        }).execute())

    def save_symbol_state(self, symbol: str, total_observed: int, recent_digits: list[int],
                           champion_weights: dict, challenger_weights: dict) -> None:
        self._safe(lambda: self.client.table("astra_symbol_state").upsert({
            "symbol": symbol, "total_observed": total_observed, "recent_digits": recent_digits,
            "champion_weights": champion_weights, "challenger_weights": challenger_weights,
            "updated_at": "now()",
        }).execute())

    def load_symbol_state(self, symbol: str) -> dict[str, Any] | None:
        result = self._safe(lambda: self.client.table("astra_symbol_state").select("*").eq("symbol", symbol).execute())
        if result and result.data:
            return result.data[0]
        return None

    def prune_old_ticks(self, retention_hours: int) -> None:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - retention_hours * 3600))
        self._safe(lambda: self.client.table("astra_ticks").delete().lt("created_at", cutoff).execute())
