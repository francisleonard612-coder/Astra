import math

from state.price_series import PriceSeries


def test_tick_log_returns_computed_correctly():
    ps = PriceSeries(symbol="R_100")
    ps.push(1000, 100.0)
    ps.push(1002, 101.0)
    ps.push(1004, 99.0)

    assert len(ps.tick_log_returns) == 2
    assert math.isclose(ps.tick_log_returns[0], math.log(101.0 / 100.0))
    assert math.isclose(ps.tick_log_returns[1], math.log(99.0 / 101.0))


def test_non_positive_price_is_skipped_not_poisoning_the_series():
    ps = PriceSeries(symbol="R_100")
    ps.push(1000, 100.0)
    ps.push(1002, 0.0)  # bad tick -- must not crash or corrupt state
    ps.push(1004, 101.0)

    assert list(ps.prices) == [100.0, 101.0]
    assert len(ps.tick_log_returns) == 1
    assert math.isclose(ps.tick_log_returns[0], math.log(101.0 / 100.0))


def test_minute_bar_close_uses_last_price_in_bucket():
    ps = PriceSeries(symbol="R_100", minute_bar_seconds=60)
    # bucket 0 (epoch 0-59): three prices, last is 102.0
    ps.push(0, 100.0)
    ps.push(10, 101.0)
    ps.push(50, 102.0)
    # bucket 60 (epoch 60-119): rolls over -- bucket 0's close (102.0) becomes
    # the baseline; no return yet, nothing to compare it against
    ps.push(65, 103.0)
    ps.push(115, 104.0)
    # bucket 120: rolls over again -- bucket 60's close (104.0) vs the
    # baseline (102.0) produces the FIRST minute return
    ps.push(120, 105.0)
    # bucket 180: one more transition -- bucket 120's close (105.0) vs the
    # new baseline (104.0) produces a second return
    ps.push(185, 106.0)

    assert len(ps.minute_log_returns) == 2
    assert math.isclose(ps.minute_log_returns[0], math.log(104.0 / 102.0))
    assert math.isclose(ps.minute_log_returns[1], math.log(105.0 / 104.0))


def test_bounded_window_evicts_oldest():
    ps = PriceSeries(symbol="R_100", max_tick_window=3)
    for i, price in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
        ps.push(i, price)
    assert len(ps.prices) == 3
    assert list(ps.prices) == [102.0, 103.0, 104.0]
