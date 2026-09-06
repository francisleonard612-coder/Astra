"""
Astra entrypoint.

Startup sequence follows spec section 51:
  1. connect to Deriv
  2. discover every R_*/1HZ* synthetic index symbol (or use ASTRA_SYMBOLS override)
  3. seed rolling state from recent tick history where available
  4. subscribe to live ticks for every symbol
  5. spawn one independent worker task per symbol (no cross-symbol blocking)
  6. each worker: observe -> update state -> predict -> evaluate -> (maybe) trade -> repeat

Monitoring dashboard/alerting (spec section 30) is intentionally out of
scope for this build -- see app/logging_setup.py.
"""
from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass

import numpy as np

from app.config import get_config
from app.logging_setup import configure_logging, get_logger
from database.repository import Repository
from database.supabase_client import make_supabase_client
from decision.decision_engine import DecisionEngine, SymbolPipeline
from execution.orders import execute_decision
from ingestion.deriv_client import DerivClient
from learning.champion_challenger import ChampionChallengerManager
from learning.retraining import RetrainingController
from research.experiment_log import ExperimentLog
from risk.risk_engine import RiskEngine
from risk.staking import StakingEngine
from state.rolling_state import StateManager

logger = get_logger("app.main")

PREDICTION_LOG_SAMPLE_EVERY_N = 20   # log a NO_TRADE prediction row this often, to keep DB volume sane
STATE_SNAPSHOT_EVERY_N_TICKS = 200
MODEL_PERF_LOG_EVERY_N_TICKS = 500
BALANCE_REFRESH_EVERY_N_TICKS = 200


@dataclass
class PendingPrediction:
    bundle: object
    predictions: dict


async def symbol_worker(symbol: str, client: DerivClient, state_manager: StateManager,
                         pipeline: SymbolPipeline, decision_engine: DecisionEngine, repo: Repository,
                         risk_engine: RiskEngine, staking: StakingEngine, retraining: RetrainingController,
                         champion_challenger: ChampionChallengerManager, cfg) -> None:
    queue = await client.subscribe_ticks(symbol)
    state = state_manager.get(symbol)
    pending: PendingPrediction | None = None
    tick_count = 0
    over_barrier = cfg.get("contracts", "over_barrier", default=2)
    under_barrier = cfg.get("contracts", "under_barrier", default=7)
    currency = cfg.currency
    duration = cfg.get("contracts", "duration", default=1)
    duration_unit = cfg.get("contracts", "duration_unit", default="t")

    log = get_logger("app.symbol_worker", symbol=symbol)
    log.info("Worker started")

    while True:
        tick = await queue.get()
        tick_count += 1

        if pending is not None:
            pipeline.observe(state, pending.bundle, pending.predictions, tick.digit, over_barrier, under_barrier)
            champion_challenger.on_trade_settled(symbol, pipeline)

        state.push(tick.digit)
        if cfg.get("database", "persist_ticks", default=True):
            repo.insert_tick(symbol, tick.epoch, tick.quote, tick.digit)

        if tick_count % BALANCE_REFRESH_EVERY_N_TICKS == 0:
            try:
                balance = await client.get_balance()
                if balance and "balance" in balance:
                    risk_engine.set_equity(float(balance["balance"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("Balance refresh failed", extra={"extra_fields": {"error": str(exc)}})

        stake = staking.current_stake(symbol)
        risk_ok, risk_reason = risk_engine.check(stake)

        decision = await decision_engine.evaluate(
            client, state, pipeline, stake=stake, currency=currency, risk_ok=risk_ok, risk_reason=risk_reason,
        )

        should_log_prediction = decision.decision != "NO_TRADE" or tick_count % PREDICTION_LOG_SAMPLE_EVERY_N == 0
        prediction_id = repo.insert_prediction(decision) if should_log_prediction else None

        if decision.decision != "NO_TRADE" and risk_ok:
            log.info("Executing trade", extra={"extra_fields": {
                "decision": decision.decision, "reason": decision.reason, "quality": decision.quality_score,
            }})
            trade_result = await execute_decision(
                client, decision, currency=currency, duration=duration, duration_unit=duration_unit,
                dry_run=cfg.dry_run,
            )
            if trade_result is not None:
                repo.insert_trade(trade_result, prediction_id)
                if trade_result.pnl is not None:
                    risk_engine.record_trade_result(trade_result.pnl)
                    staking.record_result(symbol, bool(trade_result.won))
                if trade_result.error:
                    log.warning("Trade did not settle cleanly", extra={"extra_fields": {"error": trade_result.error}})
        elif decision.decision != "NO_TRADE" and not risk_ok:
            log.info("Trade blocked by risk engine", extra={"extra_fields": {"reason": risk_reason}})
            repo.insert_risk_event(symbol, "trade_blocked", {"reason": risk_reason, "decision": decision.decision})

        retraining.maybe_retrain(symbol, pipeline.registry)

        if tick_count % STATE_SNAPSHOT_EVERY_N_TICKS == 0:
            repo.save_symbol_state(
                symbol, state.total_observed, list(state.digits)[-2000:],
                {k: v.tolist() for k, v in pipeline.champion_weights.items()},
                {k: v.tolist() for k, v in pipeline.performance.current_weights().items()},
            )

        if tick_count % MODEL_PERF_LOG_EVERY_N_TICKS == 0:
            for name in pipeline.registry.models:
                repo.insert_model_performance(
                    symbol, name, pipeline.performance.rolling_log_loss(name),
                    pipeline.champion_weights.get(name, np.zeros(10)),
                )

        # recompute a fresh prediction bundle for the *next* tick using the
        # now-updated state, and stash it so the next loop iteration can
        # score it against the actually-realized digit.
        predictions, bundle = pipeline.predict(state)
        pending = PendingPrediction(bundle=bundle, predictions=predictions)


async def discover_symbols(client: DerivClient, cfg) -> list[str]:
    if cfg.symbol_override:
        logger.info("Using ASTRA_SYMBOLS override", extra={"extra_fields": {"symbols": cfg.symbol_override}})
        return cfg.symbol_override
    prefixes = cfg.get("symbols", "prefixes", default=["R_", "1HZ"])
    symbols = await client.get_active_synthetic_symbols(prefixes)
    logger.info("Discovered symbols", extra={"extra_fields": {"count": len(symbols), "symbols": symbols}})
    return symbols


async def seed_symbol(client: DerivClient, state_manager: StateManager, symbol: str, count: int = 2000) -> None:
    try:
        history = await client.get_history(symbol, count=count)
        state_manager.seed(symbol, [t.digit for t in history])
        logger.info("Seeded symbol history", extra={"extra_fields": {"symbol": symbol, "n": len(history)}})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seeding failed, will build state from live ticks only",
                        extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})


async def main() -> None:
    cfg = get_config()
    configure_logging(cfg.log_level)
    logger.info("Starting Astra", extra={"extra_fields": {"dry_run": cfg.dry_run}})

    supabase = make_supabase_client(cfg.supabase.url, cfg.supabase.service_key, cfg.supabase.enabled)
    repo = Repository(supabase, persist_ticks=cfg.get("database", "persist_ticks", default=True))

    client = DerivClient(
        app_id=cfg.deriv.app_id, api_token=cfg.deriv.api_token,
        ws_url=cfg.deriv.ws_url, options_token_url=cfg.deriv.options_token_url,
    )
    await client.connect()
    repo.insert_system_event("app.main", "startup")

    symbols = await discover_symbols(client, cfg)
    if not symbols:
        logger.error("No symbols discovered -- nothing to trade. Check DERIV_APP_ID/token permissions.")
        return

    max_window = max(cfg.get("feature_windows", default=[2500]))
    state_manager = StateManager(max_window=max_window, max_markov_order=cfg.get("max_markov_order", default=3))

    risk_cfg = cfg.get("risk", default={})
    risk_engine = RiskEngine(
        base_stake=risk_cfg.get("base_stake", 1.0), max_stake=risk_cfg.get("max_stake", 5.0),
        max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
        max_daily_loss=risk_cfg.get("max_daily_loss", 25.0), max_drawdown=risk_cfg.get("max_drawdown", 40.0),
        max_trades_per_day=risk_cfg.get("max_trades_per_day", 500),
        cooldown_seconds_after_max_losses=risk_cfg.get("cooldown_seconds_after_max_losses", 900),
    )
    staking_cfg = risk_cfg.get("staking", {})
    staking = StakingEngine(
        base_stake=risk_cfg.get("base_stake", 1.0), enabled=staking_cfg.get("enabled", False),
        progression_factor=staking_cfg.get("progression_factor", 2.0),
        max_steps=staking_cfg.get("max_steps", 3), max_stake=risk_cfg.get("max_stake", 5.0),
    )

    retrain_cfg = cfg.get("retraining", default={})
    retraining = RetrainingController(
        every_n=retrain_cfg.get("batch_model_every_n_observations", 300),
        min_observations=retrain_cfg.get("min_observations_for_batch_models", 500),
    )

    experiment_log = ExperimentLog(repository=repo)
    cc_cfg = cfg.get("champion_challenger", default={})
    champion_challenger = ChampionChallengerManager(
        evaluate_every_n_trades=cc_cfg.get("evaluate_every_n_trades", 40),
        min_trades_to_evaluate=cc_cfg.get("min_trades_to_evaluate", 40),
        min_improvement=cc_cfg.get("min_improvement", 0.01),
        experiment_log=experiment_log,
    )

    decision_engine = DecisionEngine(cfg)

    seed_tasks = [seed_symbol(client, state_manager, s) for s in symbols]
    await asyncio.gather(*seed_tasks)

    workers = []
    for symbol in symbols:
        repo.upsert_symbol(symbol)
        pipeline = SymbolPipeline(symbol, cfg)
        workers.append(asyncio.create_task(
            symbol_worker(symbol, client, state_manager, pipeline, decision_engine, repo,
                          risk_engine, staking, retraining, champion_challenger, cfg),
            name=f"worker-{symbol}",
        ))

    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # not available on some platforms (e.g. Windows)

    await stop_event.wait()

    for w in workers:
        w.cancel()
    await client.close()
    repo.insert_system_event("app.main", "shutdown")


if __name__ == "__main__":
    asyncio.run(main())
