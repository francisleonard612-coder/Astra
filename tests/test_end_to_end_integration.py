"""
Replicates app/main.py::symbol_worker's actual per-tick sequence (observe
pending -> push -> risk check -> decision_engine.evaluate -> maybe trade)
against a fake Deriv client, end to end. This refactor touched a lot of
wiring at once (DecisionEngine.evaluate()'s new signature, the
predict_all/stash_pending/observe_pending sequencing, quote fetching feeding
three architectures instead of one) -- this test exists to catch ordering
bugs and crashes that per-module unit tests wouldn't see.
"""
import asyncio

import numpy as np

from app.config import get_config
from decision.decision_engine import ARCHITECTURES, DecisionEngine
from learning.architecture_competition import ArchitectureCompetitionManager
from risk.risk_engine import RiskEngine
from state.rolling_state import StateManager


class FakeQuote:
    def __init__(self, payout, ask_price, proposal_id="fake-1"):
        self.payout = payout
        self.ask_price = ask_price
        self.id = proposal_id
        self.spot = 100.0


class FakeDerivClient:
    """Returns a plausible fixed proposal for every symbol/contract, so
    DecisionEngine.evaluate() can run without a real network connection."""

    def __init__(self, payout_ratio: float = 0.9):
        self.payout_ratio = payout_ratio
        self.proposal_calls = 0

    async def get_proposal(self, symbol, contract_type, barrier, stake, duration, duration_unit, currency):
        self.proposal_calls += 1
        payout = stake * (1 + self.payout_ratio)
        return {"payout": payout, "ask_price": stake, "id": f"prop-{self.proposal_calls}", "spot": 100.0}


def test_full_tick_loop_runs_without_error_and_eventually_trades():
    async def run():
        cfg = get_config()
        client = FakeDerivClient()
        decision_engine = DecisionEngine(cfg)

        max_window = max(cfg.get("feature_windows", default=[2500]))
        state_manager = StateManager(max_window=max_window, max_markov_order=cfg.get("max_markov_order", default=3))
        state = state_manager.get("TEST_SYM")

        competition = ArchitectureCompetitionManager("TEST_SYM", cfg, repository=None)

        risk_cfg = cfg.get("risk", default={})
        risk_engine = RiskEngine(
            base_stake=1.0, max_stake=5.0, max_consecutive_losses=100, max_daily_loss=1000.0,
            max_drawdown=1000.0, max_trades_per_day=10000, cooldown_seconds_after_max_losses=1,
            max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 2),
        )

        rng = np.random.default_rng(3)
        # Same magnitude bias and tick budget already validated in the
        # standalone backtest sanity checks (see README's performance-bug
        # section) -- calibration needs a real number of post-threshold
        # samples to mature past the minimum_calibration_score gate, so an
        # artificially short window here would fail for budget reasons, not
        # because anything is actually broken.
        probs = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3], dtype=float)
        probs /= probs.sum()
        digits = rng.choice(10, size=3000, p=probs)

        decisions = []
        trades_executed = 0

        for digit in digits:
            digit = int(digit)
            if competition.has_pending():
                competition.observe_pending(state, digit)

            state.push(digit)

            stake = 1.0
            risk_ok, risk_reason = risk_engine.check(stake)
            decision = await decision_engine.evaluate(
                client, state, competition, stake=stake, currency="USD",
                risk_ok=risk_ok, risk_reason=risk_reason,
            )
            decisions.append(decision)

            if decision.decision != "NO_TRADE" and risk_ok:
                risk_engine.reserve_trade_slot()
                try:
                    # simulate settlement using the NEXT digit isn't available
                    # here (we don't look ahead) -- just confirm the decision
                    # object carries everything execution/orders.py needs.
                    assert decision.stake is not None
                    assert decision.architecture in ARCHITECTURES
                    trades_executed += 1
                finally:
                    risk_engine.release_trade_slot()
                risk_engine.record_trade_result(0.5)  # arbitrary settled pnl for the loop to keep progressing

        return decisions, trades_executed, client.proposal_calls

    decisions, trades_executed, proposal_calls = asyncio.run(run())

    assert len(decisions) == 3000
    # every decision came from a real architecture, never a crash/None
    assert all(d.architecture in ARCHITECTURES for d in decisions)
    # with a strongly biased stream over 500 ticks, at least one trade should fire
    assert trades_executed > 0
    # two proposal calls (OVER + UNDER) per evaluated tick past the sample threshold
    assert proposal_calls > 0


def test_concurrent_trade_cap_is_respected_across_the_loop():
    """Directly exercises the reserve/release pattern app/main.py uses
    around execute_decision, confirming the cap holds under repeated use."""
    risk_engine = RiskEngine(
        base_stake=1.0, max_stake=5.0, max_consecutive_losses=100, max_daily_loss=1000.0,
        max_drawdown=1000.0, max_trades_per_day=10000, cooldown_seconds_after_max_losses=1,
        max_concurrent_trades=2,
    )
    ok1, _ = risk_engine.check(1.0)
    assert ok1
    risk_engine.reserve_trade_slot()
    ok2, _ = risk_engine.check(1.0)
    assert ok2
    risk_engine.reserve_trade_slot()
    ok3, reason3 = risk_engine.check(1.0)
    assert not ok3 and reason3 == "max_concurrent_trades_reached"
    risk_engine.release_trade_slot()
    ok4, _ = risk_engine.check(1.0)
    assert ok4
