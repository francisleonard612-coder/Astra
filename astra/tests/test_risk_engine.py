from risk.risk_engine import RiskEngine
from risk.staking import StakingEngine


def make_engine(**overrides):
    defaults = dict(base_stake=1.0, max_stake=5.0, max_consecutive_losses=3, max_daily_loss=10.0,
                     max_drawdown=20.0, max_trades_per_day=100, cooldown_seconds_after_max_losses=1)
    defaults.update(overrides)
    return RiskEngine(**defaults)


def test_allows_trade_within_limits():
    engine = make_engine()
    ok, reason = engine.check(stake=1.0)
    assert ok
    assert reason is None


def test_blocks_stake_above_max():
    engine = make_engine()
    ok, reason = engine.check(stake=10.0)
    assert not ok
    assert reason == "stake_exceeds_max_stake"


def test_blocks_after_max_consecutive_losses():
    engine = make_engine(max_consecutive_losses=2)
    engine.record_trade_result(-1.0)
    engine.record_trade_result(-1.0)
    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "max_consecutive_losses_reached"


def test_win_resets_consecutive_loss_counter():
    engine = make_engine(max_consecutive_losses=2)
    engine.record_trade_result(-1.0)
    engine.record_trade_result(1.0)
    ok, _ = engine.check(stake=1.0)
    assert ok


def test_blocks_after_max_daily_loss():
    engine = make_engine(max_daily_loss=5.0, max_consecutive_losses=100)
    engine.record_trade_result(-6.0)
    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "max_daily_loss_reached"


def test_emergency_stop_blocks_everything():
    engine = make_engine()
    engine.trigger_emergency_stop()
    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "emergency_stop"


def test_staking_flat_when_disabled():
    staking = StakingEngine(base_stake=1.0, enabled=False, progression_factor=3.0, max_steps=3, max_stake=10.0)
    staking.record_result("R_100", won=False)
    staking.record_result("R_100", won=False)
    assert staking.current_stake("R_100") == 1.0


def test_staking_progresses_and_caps_when_enabled():
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=3.0, max_steps=2, max_stake=5.0)
    staking.record_result("R_100", won=False)
    assert staking.current_stake("R_100") == 3.0
    staking.record_result("R_100", won=False)
    # would be 9.0 (3^2) but capped at max_stake
    assert staking.current_stake("R_100") == 5.0
    staking.record_result("R_100", won=True)
    assert staking.current_stake("R_100") == 1.0
