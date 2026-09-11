"""
Trade execution.

Two things decoupled here, deliberately:

1. TradeIntent decouples "what to trade" from "how it was decided". The
   digit decision engine's `Decision` object (barrier, OVER/UNDER,
   quote_over/quote_under) and the eventual Rise/Fall decision engine
   (CALL/PUT, no barrier, a chosen duration/duration_unit per trade) are
   shaped completely differently -- TradeIntent is the common shape both
   construct and hand to OrderExecutor, so execution code doesn't need to
   know which decision engine produced it.

2. OrderExecutor.place_trade() decouples PLACING a trade from WATCHING it
   settle. A digit contract settles in 1-2 ticks (~2-4s), so awaiting
   settlement inline in the per-symbol worker loop barely mattered. A
   Rise/Fall contract can run for minutes -- awaiting it inline would stall
   tick ingestion and every other evaluation on that symbol for the entire
   contract duration. place_trade() returns as soon as Deriv accepts the
   buy; a background task (_watch_and_finalize) tracks the contract through
   to settlement and reports the result via a callback once it's known. See
   OrderExecutor.shutdown() for graceful-exit handling of tasks still
   in-flight when the process stops.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from app.logging_setup import get_logger
from ingestion.deriv_client import DerivClient, DerivRequestError
from pricing.contracts import verify_contract_direction
from pricing.payout import ContractQuote, get_quote
from risk.risk_engine import RiskEngine

logger = get_logger("execution.orders")

MAX_QUOTE_AGE_SECONDS = 2.0
DEFAULT_SETTLEMENT_TIMEOUT_SECONDS = 30.0


@dataclass
class TradeIntent:
    """What to trade, independent of which decision engine produced it."""
    symbol: str
    contract_type: str          # DIGITOVER | DIGITUNDER | CALL | PUT | CALLE | PUTE
    stake: float
    duration: int
    duration_unit: str          # "t" | "m"
    currency: str
    quote: ContractQuote        # the quote the decision was actually made against
    barrier: int | None = None  # digit contracts only; None for Rise/Fall


@dataclass
class TradeResult:
    symbol: str
    contract_type: str
    barrier: int | None
    stake: float
    payout: float
    contract_id: int | None
    won: bool | None
    pnl: float | None
    error: str | None = None


def intent_from_digit_decision(decision, duration: int, duration_unit: str, currency: str) -> TradeIntent | None:
    """Adapter preserving the current digit decision engine's behavior
    exactly, so Phase 1's execution-layer changes can land without touching
    (or waiting on) the still-digit-shaped decision engine. The eventual
    Rise/Fall decision engine will construct TradeIntent directly instead of
    going through this."""
    if decision.decision == "NO_TRADE" or decision.stake is None:
        return None
    side, barrier_str = decision.decision.split("_", 2)[1], decision.decision.rsplit("_", 1)[1]
    barrier = int(barrier_str)
    contract_type = "DIGITOVER" if side == "OVER" else "DIGITUNDER"
    quote = decision.quote_over if side == "OVER" else decision.quote_under
    if quote is None:
        return None
    return TradeIntent(
        symbol=decision.symbol, contract_type=contract_type, stake=decision.stake,
        duration=duration, duration_unit=duration_unit, currency=currency,
        quote=quote, barrier=barrier,
    )


def intent_from_rise_fall_decision(decision, currency: str) -> TradeIntent | None:
    """Adapter mirroring intent_from_digit_decision, for
    decision/rise_fall_decision_engine.py's RiseFallDecision. Unlike the
    digit decision (which needs duration/duration_unit passed in from
    config, since digit contracts always trade at one fixed duration),
    RiseFallDecision already carries its own duration/duration_unit and
    quote -- MC duration selection happens per-decision there, not fixed
    per-deployment."""
    if decision.decision == "NO_TRADE" or decision.stake is None or decision.quote is None:
        return None
    return TradeIntent(
        symbol=decision.symbol, contract_type=decision.contract_type, stake=decision.stake,
        duration=decision.duration, duration_unit=decision.duration_unit, currency=currency,
        quote=decision.quote, barrier=None,
    )


class OrderExecutor:
    def __init__(self, client: DerivClient, repo, risk_engine: RiskEngine, dry_run: bool,
                 settlement_timeout_seconds: float = DEFAULT_SETTLEMENT_TIMEOUT_SECONDS):
        self.client = client
        self.repo = repo
        self.risk_engine = risk_engine
        self.dry_run = dry_run
        self.settlement_timeout_seconds = settlement_timeout_seconds
        self._pending_tasks: set[asyncio.Task] = set()

    async def place_trade(self, intent: TradeIntent, prediction_id: int | None = None,
                           on_settled: Callable[[TradeResult], None] | None = None) -> TradeResult:
        """Caller contract: call risk_engine.reserve_trade_slot() synchronously
        (see its docstring) immediately before calling this. This method
        takes ownership of releasing that slot from here on -- it releases
        it itself if the buy never goes through, or hands off the release
        to the background settlement watcher if one gets scheduled. Callers
        never need to release the slot themselves after calling this.

        Returns a TradeResult with won=None/pnl=None (settlement pending,
        handled by on_settled once known) on a successful buy, or an
        immediate failure TradeResult if no contract was ever bought.
        """
        try:
            return await self._place_trade_inner(intent, prediction_id, on_settled)
        except Exception:
            self.risk_engine.release_trade_slot()
            raise

    async def _place_trade_inner(self, intent: TradeIntent, prediction_id: int | None,
                                  on_settled: Callable[[TradeResult], None] | None) -> TradeResult:
        # Re-validate: fetch a fresh quote immediately before buying and
        # refuse to trade on stale contract information (spec section 18).
        fresh_quote = await get_quote(
            self.client, intent.symbol, intent.contract_type, intent.barrier, intent.stake,
            intent.duration, intent.duration_unit, intent.currency, bypass_cache=True,
        )
        if fresh_quote is None:
            self.risk_engine.release_trade_slot()
            return self._fail(intent, 0.0, "quote_unavailable_at_execution", prediction_id)

        payout_drift = abs(fresh_quote.payout - intent.quote.payout) / max(intent.quote.payout, 1e-9)
        if payout_drift > 0.15:
            logger.warning("Payout drifted too much between decision and execution, skipping",
                            extra={"extra_fields": {"symbol": intent.symbol, "drift": payout_drift}})
            self.risk_engine.release_trade_slot()
            return self._fail(intent, fresh_quote.payout, "stale_quote_payout_drift", prediction_id)

        if self.dry_run:
            logger.info("DRY_RUN: would execute trade", extra={"extra_fields": {
                "symbol": intent.symbol, "contract_type": intent.contract_type, "barrier": intent.barrier,
                "stake": intent.stake, "payout": fresh_quote.payout,
            }})
            self.risk_engine.release_trade_slot()
            result = TradeResult(intent.symbol, intent.contract_type, intent.barrier, intent.stake,
                                  fresh_quote.payout, None, None, 0.0, error=None)
            self.repo.insert_trade(result, prediction_id)
            # pnl=0.0 here (not None) is deliberate, matching a real
            # settled trade's bookkeeping call below -- dry_run should
            # exercise trades_today/consecutive_losses tracking exactly
            # like a live trade would, just with a neutral outcome, so
            # max_trades_per_day and similar gates behave identically
            # whether or not the account is live.
            self.risk_engine.record_trade_result(0.0)
            return result

        if not verify_contract_direction({"longcode": fresh_quote.longcode}, intent.contract_type):
            logger.error("Contract longcode disagrees with expected Rise/Fall direction -- "
                         "refusing to trade until this is verified",
                         extra={"extra_fields": {"symbol": intent.symbol, "contract_type": intent.contract_type}})
            self.risk_engine.release_trade_slot()
            return self._fail(intent, fresh_quote.payout, "direction_mismatch_tripwire", prediction_id)

        try:
            buy_resp = await self.client.buy(fresh_quote.proposal_id, fresh_quote.ask_price)
        except DerivRequestError as exc:
            logger.error("Buy failed", extra={"extra_fields": {"symbol": intent.symbol, "error": str(exc)}})
            self.risk_engine.release_trade_slot()
            return self._fail(intent, fresh_quote.payout, str(exc), prediction_id)

        contract_id = buy_resp.get("contract_id")
        if contract_id is None:
            self.risk_engine.release_trade_slot()
            return self._fail(intent, fresh_quote.payout, "buy_response_missing_contract_id", prediction_id)

        result = TradeResult(intent.symbol, intent.contract_type, intent.barrier, intent.stake,
                              fresh_quote.payout, contract_id, None, None, error=None)
        task = asyncio.create_task(
            self._watch_and_finalize(intent, contract_id, fresh_quote.payout, prediction_id, on_settled),
            name=f"settle-{intent.symbol}-{contract_id}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return result

    def _fail(self, intent: TradeIntent, payout: float, error: str, prediction_id: int | None) -> TradeResult:
        # Persisted here, not left to the caller: every OTHER terminal
        # TradeResult (the dry-run path above, the async-settled path in
        # _watch_and_finalize) also persists itself, so a caller checking
        # `if result.contract_id is not None` to decide whether to log it
        # would silently drop every failed-buy attempt from astra_trades --
        # exactly the record you want when diagnosing why trades aren't
        # landing.
        result = TradeResult(intent.symbol, intent.contract_type, intent.barrier, intent.stake,
                              payout, None, None, None, error=error)
        self.repo.insert_trade(result, prediction_id)
        return result

    async def _watch_and_finalize(self, intent: TradeIntent, contract_id: int, payout: float,
                                   prediction_id: int | None,
                                   on_settled: Callable[[TradeResult], None] | None) -> None:
        # A minute-duration contract needs a settlement wait longer than the
        # contract itself, not the flat 30s that was fine when every
        # contract was a 1-2 tick digit trade -- otherwise every minute
        # contract would time out before it could ever settle.
        timeout = max(
            self.settlement_timeout_seconds,
            intent.duration * (65.0 if intent.duration_unit == "m" else 4.0),
        )
        try:
            settled = await self.client.wait_for_contract_settlement(contract_id, timeout=timeout)
            profit = settled.get("profit")
            won = pnl = None
            error = None
            if profit is not None:
                pnl = float(profit)
                won = pnl > 0
            else:
                error = "settlement_timeout_or_unknown"
            result = TradeResult(intent.symbol, intent.contract_type, intent.barrier, intent.stake,
                                  payout, contract_id, won, pnl, error=error)
        except Exception as exc:  # noqa: BLE001
            logger.error("Settlement watcher crashed", exc_info=exc, extra={"extra_fields": {
                "symbol": intent.symbol, "contract_id": contract_id,
            }})
            result = TradeResult(intent.symbol, intent.contract_type, intent.barrier, intent.stake,
                                  payout, contract_id, None, None, error=str(exc))
        finally:
            self.risk_engine.release_trade_slot()

        self.repo.insert_trade(result, prediction_id)
        if result.pnl is not None:
            self.risk_engine.record_trade_result(result.pnl)
        if on_settled is not None:
            try:
                on_settled(result)
            except Exception:  # noqa: BLE001
                logger.error("on_settled callback failed", extra={"extra_fields": {"symbol": intent.symbol}})

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Give in-flight settlement watchers a bounded chance to finish (and
        persist their result) before the process exits, instead of letting
        asyncio silently drop them -- which would lose the DB record of a
        contract that's genuinely still open on Deriv's side."""
        if not self._pending_tasks:
            return
        logger.info("Waiting for in-flight settlements before shutdown", extra={"extra_fields": {
            "pending": len(self._pending_tasks),
        }})
        done, pending = await asyncio.wait(self._pending_tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            logger.warning("Some settlement watchers did not finish before shutdown", extra={"extra_fields": {
                "abandoned": len(pending),
            }})
