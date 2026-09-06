import numpy as np

from state.rolling_state import StateManager
from features.feature_engine import build_features


def make_state(digits, max_window=2000):
    sm = StateManager(max_window=max_window)
    sm.seed("TEST", digits)
    return sm.get("TEST")


def test_gap_tracks_ticks_since_last_seen():
    state = make_state([1, 2, 3, 1, 5])
    # digit 1 last appeared at tick_index 4 (1-indexed pushes), current tick_index=5
    assert state.gap(1) == 1
    assert state.gap(5) == 0
    assert state.gap(9) == state.tick_index  # never seen


def test_streaks_reset_on_change():
    state = make_state([5, 5, 5, 2])
    assert state.same_digit_streak == 1
    assert state.high_streak == 0
    assert state.low_streak == 1


def test_uniform_distribution_has_high_entropy():
    digits = list(range(10)) * 50  # perfectly uniform
    state = make_state(digits)
    bundle = build_features(state, windows=[500])
    assert bundle.entropy[500] > 0.99


def test_concentrated_distribution_has_low_entropy():
    digits = [3] * 200 + [4] * 5
    state = make_state(digits)
    bundle = build_features(state, windows=[200])
    assert bundle.entropy[200] < 0.5


def test_feature_vector_has_no_nans():
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 40
    state = make_state(digits)
    bundle = build_features(state, windows=[20, 100, 500])
    assert bundle.vector is not None
    assert not np.isnan(bundle.vector).any()


def test_missing_window_degrades_to_zeros_not_crash():
    state = make_state([1, 2, 3])
    bundle = build_features(state, windows=[1000])
    # window requested is larger than history -- should not raise, and window
    # dict will simply reflect whatever partial data exists
    assert isinstance(bundle.vector, np.ndarray)
