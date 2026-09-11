import numpy as np

from features.feature_engine import build_features
from models.digit_specialist import DigitSpecialist, DigitSpecialistArchitecture
from state.rolling_state import StateManager


def test_specialist_starts_at_fair_prior():
    s = DigitSpecialist(digit=7)
    assert abs(s.bayes_probability() - 0.1) < 1e-9


def test_specialist_bayes_probability_moves_toward_observed_rate():
    s = DigitSpecialist(digit=3)
    for _ in range(50):
        s.observe(feature_vector=None, actual_digit=3)  # always occurs
    for _ in range(10):
        s.observe(feature_vector=None, actual_digit=9)  # never occurs
    # observed rate is 50/60 ~ 0.83, prior was 0.1 -- should have moved a lot
    assert s.bayes_probability() > 0.5


def test_architecture_output_sums_to_one():
    arch = DigitSpecialistArchitecture()
    sm = StateManager(max_window=500)
    sm.seed("X", list(np.random.default_rng(0).integers(0, 10, size=200)))
    state = sm.get("X")
    bundle = build_features(state, windows=[100])
    vec = arch.predict(state, bundle)
    assert abs(vec.sum() - 1.0) < 1e-9
    assert (vec > 0).all()


def test_architecture_learns_a_biased_digit():
    arch = DigitSpecialistArchitecture()
    sm = StateManager(max_window=1000)
    state = sm.get("X")
    rng = np.random.default_rng(1)
    probs = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 5], dtype=float)
    probs /= probs.sum()
    digits = list(rng.choice(10, size=600, p=probs))

    for d in digits:
        state.push(d)
        bundle = build_features(state, windows=[500])
        arch.observe(bundle, d)

    bundle = build_features(state, windows=[500])
    vec = arch.predict(state, bundle)
    # digit 9 was heavily overrepresented -- the specialist for digit 9
    # should now assign it noticeably more than the fair 1/10
    assert vec[9] > 0.15


def test_components_are_exposed_separately_for_agreement_scoring():
    arch = DigitSpecialistArchitecture()
    sm = StateManager(max_window=200)
    sm.seed("X", [1, 2, 3, 4, 5] * 20)
    state = sm.get("X")
    bundle = build_features(state, windows=[100])
    bayes_vec, logit_vec, blended = arch.predict_components(bundle)
    for v in (bayes_vec, logit_vec, blended):
        assert abs(v.sum() - 1.0) < 1e-9
