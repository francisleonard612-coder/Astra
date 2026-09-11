import asyncio

import numpy as np

import pricing.payout as payout_module
from decision.rise_fall_decision_engine import DRIFT_STAKE_REDUCTION, RiseFallSymbolPipeline, summary_reason
from pricing.contracts import FALL, RISE


def _reset_payout_module_state():
    payout_module._quote_cache.clear()
    payout_module._rate_limit_cooldown_until = 0.0
    payout_module._consecutive_rate_limit_failures = 0


class FakeDerivClient:
    def __init__(self, payout_for_contract_type: dict[str, float]):
        self.payout_for_contract_type = payout_for_contract_type
        self.calls = []

    async def get_proposal(self, symbol, contract_type, barrier, stake, duration, duration_unit, currency):
        self.calls.append((contract_type, duration, duration_unit))
        assert barrier is None
        return {
            "payout": self.payout_for_contract_type.get(contract_type, 2.0),
            "ask_price": stake, "id": f"prop-{contract_type}-{duration}{duration_unit}",
            "spot": 100.0, "longcode": "rises" if contract_type == RISE else "falls",
        }


def _seed_trending_history(pipeline: RiseFallSymbolPipeline, seed: int, drift: float) -> None:
    rng = np.random.default_rng(seed)
    tick_returns = rng.normal(drift, 0.001, size=300)
    minute_returns = rng.normal(drift, 0.002, size=300)
    pipeline.price_series.tick_log_returns.extend(tick_returns.tolist())
    pipeline.price_series.minute_log_returns.extend(minute_returns.tolist())


def test_no_trade_with_no_history_at_all():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0)
        client = FakeDerivClient({RISE: 2.0, FALL: 2.0})
        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=500)
        assert decision.decision == "NO_TRADE"
        assert decision.contract_type is None
        assert client.calls == []  # never even reached a real quote

    asyncio.run(run())


def test_neutral_regime_skips_before_any_quote_is_fetched():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0)
        client = FakeDerivClient({RISE: 2.0, FALL: 2.0})
        # seed=2 confirmed (via direct hurst_rs/classify_regime check) to
        # land in the NEUTRAL band with drift=0.0 -- a single MC/Hurst
        # realization on 300 noise points is NOT guaranteed to read exactly
        # 0.5 (finite-sample R/S Hurst has a well-documented positive bias;
        # see test_hurst_volatility.py), so this must be a seed verified to
        # land there, not an arbitrary one.
        _seed_trending_history(pipeline, seed=2, drift=0.0)
        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=500)
        assert decision.decision == "NO_TRADE"
        assert decision.regime == "NEUTRAL"
        assert client.calls == []  # regime gate stops it before any network call

    asyncio.run(run())


def test_no_edge_clears_on_average_under_fair_pricing_and_pure_noise():
    """A SINGLE noise realization's recent-window sample mean can legitimately
    look like real drift (same phenomenon characterized in
    test_monte_carlo_duration.py's v10 bias test) -- asserting "no trade"
    for one arbitrary seed isn't well-posed. What should hold, and is worth
    asserting, is the AVERAGE across many independent histories.

    The threshold below (30%, not the naively-expected ~95%) reflects a
    real finding from measuring this directly, documented in this module's
    own docstring under "MULTIPLE-COMPARISONS WARNING": evaluate() checks
    FOUR candidates per cycle and takes the best edge, which measurably
    inflates the false-positive rate above what a single min_edge=0.03
    threshold implies for one comparison. This test exists to catch a
    REGRESSION in that already-elevated rate (e.g. a change that makes it
    worse still), not to assert the rate is small -- it measurably isn't,
    and that's flagged prominently rather than hidden behind a loose
    threshold here.
    """
    _reset_payout_module_state()

    async def run():
        no_trade_count = 0
        n_seeds = 20
        for seed in range(n_seeds):
            pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
            client = FakeDerivClient({RISE: 2.0, FALL: 2.0})  # fairly priced, breakeven 50/50
            _seed_trending_history(pipeline, seed=seed, drift=0.0)
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=1500,
                                                rng=np.random.default_rng(seed))
            if decision.decision == "NO_TRADE":
                no_trade_count += 1
        assert no_trade_count >= n_seeds * 0.3

    asyncio.run(run())


def test_dramatic_mispricing_produces_a_trade_regardless_of_regime_noise():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        # FALL is dramatically underpriced by Deriv (breakeven ~2%) relative
        # to any realistic win probability -- should surface as the best
        # edge, decisively.
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        # seed=0 confirmed (via direct hurst_rs/classify_regime check) to
        # land in a TRADEABLE regime (TREND_QUIET) with this drift setup --
        # an earlier version of this test used seed=7 with a conditional
        # "if decision.decision != NO_TRADE" guard and passed VACUOUSLY,
        # since seed=7 actually lands in NEUTRAL (see test_hurst_volatility.py
        # for why finite-sample Hurst varies this much seed to seed) and the
        # guarded assertions never ran. Using a confirmed-tradeable seed
        # with unconditional assertions catches that class of mistake.
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))

        assert decision.decision != "NO_TRADE"
        assert decision.contract_type == FALL
        assert decision.stake is not None and decision.stake > 0
        assert decision.edge is not None and decision.edge > 0.3
        assert decision.duration_unit in ("t", "m")

    asyncio.run(run())


def test_record_outcome_calibrates_only_the_traded_candidate_rise():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    pipeline._pending = (RISE, 0.8)

    pipeline.record_outcome(won=True)

    # RISE won -> price went up -> outcome=1 for RISE's own calibration
    assert pipeline.calibration[RISE]._outcome[-1] == 1
    assert pipeline.calibration[RISE]._raw[-1] == 0.8
    # FALL was NOT the traded candidate this cycle -- untouched
    assert len(pipeline.calibration[FALL]._outcome) == 0
    assert pipeline._pending is None


def test_record_outcome_calibrates_only_the_traded_candidate_fall():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    pipeline._pending = (FALL, 0.7)

    pipeline.record_outcome(won=True)

    # FALL won -> price went DOWN -> outcome=1 for FALL's own calibration
    # (FALL's raw_prob was a probability of price falling, and it did)
    assert pipeline.calibration[FALL]._outcome[-1] == 1
    assert pipeline.calibration[FALL]._raw[-1] == 0.7
    assert len(pipeline.calibration[RISE]._outcome) == 0


def test_record_outcome_with_no_pending_candidate_only_feeds_cusum():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    stat_before = pipeline.drift._cusum_stat
    pipeline.record_outcome(won=False)  # no evaluate() ran first -- nothing pending
    assert pipeline.drift._cusum_stat > stat_before
    assert len(pipeline.calibration[RISE]._outcome) == 0
    assert len(pipeline.calibration[FALL]._outcome) == 0


def test_record_outcome_always_feeds_cusum():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    pipeline._pending = (RISE, 0.6)
    stat_before = pipeline.drift._cusum_stat
    pipeline.record_outcome(won=False)
    assert pipeline.drift._cusum_stat > stat_before  # a loss pushes CUSUM up


def test_evaluate_stashes_pending_state_only_for_the_winning_candidate():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # confirmed tradeable regime

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert pipeline._pending is not None
        pending_type, pending_prob = pipeline._pending
        assert pending_type == decision.contract_type
        assert pending_prob == decision.mc_win_probability

    asyncio.run(run())


def test_cancel_pending_clears_without_recording_anything():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    pipeline._pending = (RISE, 0.8)
    stat_before = pipeline.drift._cusum_stat

    pipeline.cancel_pending()

    assert pipeline._pending is None
    assert len(pipeline.calibration[RISE]._outcome) == 0  # nothing recorded
    assert pipeline.drift._cusum_stat == stat_before  # CUSUM untouched


def test_cancel_pending_unblocks_the_pending_slot_guard():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # confirmed tradeable regime

        first = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                         rng=np.random.default_rng(0))
        assert first.decision != "NO_TRADE"
        assert pipeline._pending is not None

        # simulate the buy failing (stale quote, rejected, whatever) --
        # caller cancels the pending decision rather than settling it
        pipeline.cancel_pending()
        assert pipeline._pending is None

        # a later evaluate() call must be free to trade again, not
        # permanently locked out by a decision that never actually settled
        second = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                          rng=np.random.default_rng(0))
        assert "still awaiting settlement" not in second.reason

    asyncio.run(run())


def test_evaluate_refuses_a_new_trade_while_one_is_still_pending_settlement():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # confirmed tradeable regime

        first = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                         rng=np.random.default_rng(0))
        assert first.decision != "NO_TRADE"
        assert pipeline._pending is not None
        pending_before = pipeline._pending

        # a second evaluate() call, simulating the worker loop running again
        # while the first trade is still in flight -- must NOT overwrite
        # the pending slot or produce a new trade
        second = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                          rng=np.random.default_rng(1))
        assert second.decision == "NO_TRADE"
        assert "still awaiting settlement" in second.reason
        assert pipeline._pending == pending_before  # untouched

    asyncio.run(run())


def test_drift_degraded_quote_matches_the_reduced_stake_basis():
    """Regression test for a real bug found while wiring this: if the
    decision's quote stayed at base_stake while decision.stake was the
    drift-reduced amount, OrderExecutor's fresh pre-buy quote (fetched at
    the reduced stake) would look like a ~50% payout mismatch against it
    (payout scales with stake) and reject every drift-reduced trade as
    stale. The quote attached to the decision must be fetched at the SAME
    stake as decision.stake."""
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=2.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # confirmed tradeable regime
        pipeline.drift.check_all = lambda *a, **k: True

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert decision.quote is not None
        assert decision.quote.ask_price == decision.stake  # same basis, not base_stake

    asyncio.run(run())


def test_drift_degraded_halves_the_stake():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=2.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # confirmed tradeable regime

        # force the drift detector to report degraded regardless of actual input
        pipeline.drift.check_all = lambda *a, **k: True

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert decision.drift_degraded is True
        assert decision.stake == round(2.0 * DRIFT_STAKE_REDUCTION, 2)

    asyncio.run(run())


def test_summary_reason_categorizes_regime_skip():
    from decision.rise_fall_decision_engine import RiseFallDecision
    decision = RiseFallDecision("1HZ10V", "NO_TRADE", "NEUTRAL", "H=0.50 in neutral band...")
    assert summary_reason(decision) == "regime:NEUTRAL"


def test_summary_reason_categorizes_pending_settlement():
    from decision.rise_fall_decision_engine import RiseFallDecision
    decision = RiseFallDecision("1HZ10V", "NO_TRADE", "TREND_QUIET", "previous trade still awaiting settlement")
    assert summary_reason(decision) == "pending_settlement"


def test_summary_reason_categorizes_insufficient_edge():
    from decision.rise_fall_decision_engine import RiseFallDecision
    decision = RiseFallDecision("1HZ10V", "NO_TRADE", "TREND_QUIET",
                                 "H=0.60 trending; no candidate cleared min_edge=0.03")
    assert summary_reason(decision) == "insufficient_edge"
