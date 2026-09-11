"""
Runs the ACTUAL app/main.py::symbol_worker coroutine (not a reimplementation)
against a fake Deriv client and repository, feeding it a real tick queue.
This is the test that catches wiring bugs -- wrong argument order, a
misspelled attribute, a mismatched call signature -- that unit tests of the
individual pieces (decision engine, order executor, calibration, ...) can't
see, because each of those is tested in isolation with its own fakes.
"""
import asyncio

import numpy as np

import pricing.payout as payout_module
from app.config import get_config
from app.main import symbol_worker
from database.repository import _json_safe
from decision.rise_fall_decision_engine import RiseFallSymbolPipeline
from execution.orders import OrderExecutor
from ingestion.deriv_client import Tick
from pricing.contracts import FALL, RISE
from risk.risk_engine import RiskEngine


def _reset_payout_module_state():
    payout_module._quote_cache.clear()
    payout_module._rate_limit_cooldown_until = 0.0
    payout_module._consecutive_rate_limit_failures = 0


class FakeDerivClient:
    """FALL is dramatically underpriced relative to any real win probability
    -- combined with genuine bearish drift seeded into the pipeline, this
    should reliably produce a FALL trade within a handful of evaluations,
    without relying on MC estimation noise the way a fair-pricing test
    would (see test_rise_fall_decision_engine.py's multiple-comparisons
    finding for why that would be flaky here)."""

    def __init__(self):
        self.proposal_calls = 0
        self.buy_calls = 0
        self.settlement_calls = 0

    async def get_proposal(self, symbol, contract_type, barrier, stake, duration, duration_unit, currency):
        self.proposal_calls += 1
        assert barrier is None
        payout = 50.0 if contract_type == FALL else 1.5
        return {"payout": payout, "ask_price": stake, "id": f"prop-{self.proposal_calls}",
                "spot": 100.0, "longcode": "falls" if contract_type == FALL else "rises"}

    async def buy(self, proposal_id, price):
        self.buy_calls += 1
        return {"contract_id": self.buy_calls}

    async def wait_for_contract_settlement(self, contract_id, timeout=30.0):
        self.settlement_calls += 1
        # FALL contracts (the ones actually worth taking here) settle as
        # wins; anything else settles as a loss -- doesn't matter which for
        # this test, only that settlement completes and the callback chain
        # runs without crashing.
        return {"is_sold": 1, "profit": 5.0}

    async def get_balance(self):
        return {"balance": 1000.0}


class FakeRepository:
    def __init__(self):
        self.ticks = []
        self.trades = []
        self.risk_events = []
        self.system_events = []

    def insert_tick(self, symbol, epoch, quote, digit):
        self.ticks.append((symbol, epoch, quote, digit))

    def insert_trade(self, trade, prediction_id=None):
        self.trades.append(_json_safe(trade.__dict__))

    def insert_risk_event(self, symbol, event_type, detail):
        self.risk_events.append((symbol, event_type, detail))

    def insert_system_event(self, component, event_type, detail=None):
        self.system_events.append((component, event_type, detail))


def test_symbol_worker_runs_real_ticks_through_to_a_settled_trade():
    _reset_payout_module_state()

    async def run():
        cfg = get_config()
        client = FakeDerivClient()
        repo = FakeRepository()
        risk_engine = RiskEngine(
            base_stake=1.0, max_stake=5.0, max_consecutive_losses=100, max_daily_loss=1000.0,
            max_drawdown=1000.0, max_trades_per_day=10000, cooldown_seconds_after_max_losses=1,
            max_concurrent_trades=2,
        )
        order_executor = OrderExecutor(client, repo, risk_engine, dry_run=False)

        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        # Pre-seed with genuine bearish drift, same construction as the
        # decision-engine tests, so evaluate() has enough history to
        # classify a regime and find the FALL mispricing from tick zero of
        # the live loop rather than needing hundreds of live ticks first.
        # seed=0 confirmed (see test_rise_fall_decision_engine.py) to land
        # in a tradeable regime with this drift setup.
        rng = np.random.default_rng(0)
        tick_returns = rng.normal(-0.01, 0.001, size=300)
        minute_returns = rng.normal(-0.01, 0.002, size=300)
        pipeline.price_series.tick_log_returns.extend(tick_returns.tolist())
        pipeline.price_series.minute_log_returns.extend(minute_returns.tolist())
        # observe_tick needs real prices in the deque too (evaluate() itself
        # only reads the log-return deques, but seed prices for realism)
        price = 100.0
        for i, r in enumerate(tick_returns):
            price *= np.exp(r)
            pipeline.price_series.prices.append(price)

        queue = asyncio.Queue()

        async def fake_subscribe_ticks(symbol, queue_size=2000):
            return queue

        client.subscribe_ticks = fake_subscribe_ticks

        worker_task = asyncio.create_task(
            symbol_worker("1HZ10V", client, pipeline, repo, risk_engine, order_executor, cfg)
        )

        # feed real ticks and let the worker process them
        for i in range(30):
            queue.put_nowait(Tick(symbol="1HZ10V", epoch=1_700_000_000 + i * 2, quote=100.0 + i * 0.01,
                                   digit=i % 10))
            await asyncio.sleep(0.01)

        # let any scheduled settlement-watcher background tasks finish too
        await asyncio.sleep(0.1)

        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        await order_executor.shutdown(timeout=2.0)

        # the worker ran 30 real ticks through the real code path without
        # crashing, and produced real, persisted output
        assert len(repo.ticks) == 30
        assert client.proposal_calls > 0
        # seed=0's dramatic FALL mispricing should reliably produce a trade
        # within 30 evaluations -- assert the full buy -> settle -> persist
        # chain actually ran, not just that it would have if it had
        assert client.buy_calls > 0
        assert client.settlement_calls > 0
        assert len(repo.trades) > 0
        settled_trades = [t for t in repo.trades if t["contract_id"] is not None]
        assert len(settled_trades) > 0
        for t in settled_trades:
            assert t["error"] is None
            assert t["won"] is True
            assert t["barrier"] is None  # Rise/Fall trades never carry a barrier
        # the risk engine's slot accounting must have unwound cleanly --
        # nothing left "open" after every settlement watcher completed
        assert risk_engine.state.open_trades == 0

    asyncio.run(run())
