import numpy as np

from learning.online import PerformanceTracker
from models.ensemble import combine

INITIAL = {"good_at_5": 0.5, "good_at_others": 0.5}


def _confident_vector(favored_digit: int, confidence: float = 0.9) -> np.ndarray:
    vec = np.full(10, (1 - confidence) / 9)
    vec[favored_digit] = confidence
    return vec


def test_weights_sum_to_one_per_digit():
    tracker = PerformanceTracker(initial_weights=INITIAL, min_weight=0.02, window=500)
    weights = tracker.current_weights()
    for d in range(10):
        total = sum(w[d] for w in weights.values())
        assert abs(total - 1.0) < 1e-9


def test_model_specialized_on_one_digit_gets_upweighted_there_and_not_elsewhere():
    tracker = PerformanceTracker(initial_weights=INITIAL, min_weight=0.02, window=500)

    # "good_at_5" nails digit 5 whenever it occurs and is mediocre (near-uniform)
    # otherwise. "good_at_others" is consistently decent everywhere but never
    # exceptional. Feed 200 rounds where the actual digit cycles 0..9.
    rng = np.random.default_rng(0)
    for i in range(400):
        actual = i % 10
        if actual == 5:
            preds = {
                "good_at_5": _confident_vector(5, 0.95),
                "good_at_others": np.full(10, 0.1),  # uniform -- no opinion
            }
        else:
            preds = {
                "good_at_5": np.full(10, 0.1),  # uniform -- no opinion on non-5 digits
                "good_at_others": _confident_vector(actual, 0.5),  # moderately confident, correct
            }
        tracker.record(preds, actual)

    weights = tracker.current_weights()
    # on digit 5, the specialist should dominate
    assert weights["good_at_5"][5] > weights["good_at_others"][5]
    # on a non-5 digit, the generalist should dominate instead
    assert weights["good_at_others"][2] > weights["good_at_5"][2]


def test_insufficient_history_falls_back_to_initial_weight():
    tracker = PerformanceTracker(initial_weights=INITIAL, min_weight=0.02, window=500)
    # only a handful of observations -- below _MIN_HISTORY
    for i in range(5):
        tracker.record({"good_at_5": _confident_vector(5), "good_at_others": _confident_vector(i % 10)}, i % 10)
    weights = tracker.current_weights()
    for d in range(10):
        assert abs(weights["good_at_5"][d] - weights["good_at_others"][d]) < 1e-6  # still ~equal (both at initial)


def test_combine_accepts_both_scalar_and_per_digit_weights():
    preds = {"a": np.full(10, 0.1), "b": np.array([0.0] * 9 + [1.0])}

    scalar_result = combine(preds, {"a": 0.5, "b": 0.5})
    assert abs(scalar_result.sum() - 1.0) < 1e-9

    # two per-digit configs, identical everywhere except how much weight
    # digit 9 gives to "b" vs "a" -- isolates the per-digit effect from
    # the unrelated renormalization-baseline differences a full scalar-vs-
    # vector comparison would introduce.
    low_trust_in_b = {"a": np.full(10, 0.9), "b": np.full(10, 0.1)}
    high_trust_in_b = {"a": np.full(10, 0.9), "b": np.full(10, 0.1)}
    high_trust_in_b["b"] = high_trust_in_b["b"].copy()
    high_trust_in_b["a"] = high_trust_in_b["a"].copy()
    high_trust_in_b["b"][9] = 0.9
    high_trust_in_b["a"][9] = 0.1

    low_result = combine(preds, low_trust_in_b)
    high_result = combine(preds, high_trust_in_b)
    assert abs(low_result.sum() - 1.0) < 1e-9
    assert abs(high_result.sum() - 1.0) < 1e-9
    # trusting "b" more specifically on digit 9 should pull digit 9's
    # combined probability higher (toward b's confident 1.0 there)
    assert high_result[9] > low_result[9]
