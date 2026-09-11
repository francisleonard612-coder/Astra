import numpy as np

from regime.hurst_volatility import hurst_rs, realized_vol_and_baseline


def test_hurst_returns_neutral_with_too_little_history():
    assert hurst_rs(np.zeros(10)) == 0.5
    assert hurst_rs([]) == 0.5


def test_hurst_on_prices_vs_returns_reproduces_documented_bug():
    """The exact scenario the source's own docstring documents: Hurst
    computed on raw (random-walk) prices spuriously reads ~1.0 regardless
    of true dynamics, because prices never mean-revert. On the log-returns
    of the SAME series, it correctly reads near 0.5 (no information) for a
    genuine random walk."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.001, size=2000)
    prices = 100 * np.exp(np.cumsum(returns))

    h_on_returns = hurst_rs(returns)
    h_on_prices = hurst_rs(prices)  # wrong input type -- reproduces the bug

    assert 0.35 < h_on_returns < 0.65  # near 0.5, correctly "no information"
    assert h_on_prices > 0.95  # spuriously reads maximal trending


def test_hurst_detects_persistent_trending_series():
    rng = np.random.default_rng(1)
    # positively autocorrelated (trending) returns: each step nudged toward
    # the sign of the previous step
    returns = [rng.normal(0, 0.001)]
    for _ in range(1999):
        returns.append(0.6 * np.sign(returns[-1]) * abs(rng.normal(0, 0.001)) + rng.normal(0, 0.0003))
    h = hurst_rs(np.array(returns))
    assert h > 0.55


def test_realized_vol_and_baseline_zero_with_insufficient_history():
    now, baseline = realized_vol_and_baseline(np.zeros(10))
    assert now == 0.0
    assert baseline == 0.0


def test_realized_vol_and_baseline_computed_with_enough_history():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.002, size=300)
    now, baseline = realized_vol_and_baseline(returns)
    assert now > 0.0
    assert baseline > 0.0
    # both should be roughly the same order of magnitude as the true vol
    assert 0.0005 < now < 0.01
    assert 0.0005 < baseline < 0.01


def test_realized_vol_now_reflects_a_recent_spike_baseline_does_not_yet():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 0.001, size=250)
    spike = rng.normal(0.0, 0.01, size=30)  # sudden 10x vol spike, very recent
    returns = np.concatenate([calm, spike])
    now, baseline = realized_vol_and_baseline(returns, now_window=30, baseline_window=200)
    assert now > baseline * 2  # the recent spike shows up in "now" well before it dominates the baseline
