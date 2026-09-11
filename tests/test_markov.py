import numpy as np

from features.feature_engine import build_features
from models.markov import MarkovModel
from state.rolling_state import StateManager


def test_markov_predicts_deterministic_pattern():
    # digit after "1,2" is always 3 -- a long, clean repeating pattern
    pattern = [1, 2, 3] * 200
    sm = StateManager(max_window=2000)
    sm.seed("TEST", pattern)
    state = sm.get("TEST")
    bundle = build_features(state, windows=[2000])

    model = MarkovModel(max_order=2)
    probs = model.predict(state, bundle)
    assert probs.argmax() == 1
    assert probs[1] > 0.9


def test_markov_backs_off_when_context_too_rare():
    # mostly random-looking history with one single occurrence of a rare
    # 3-digit context right at the end -- not enough support to trust order-3
    np.random.seed(0)
    digits = list(np.random.randint(0, 10, size=500))
    sm = StateManager(max_window=2000)
    sm.seed("TEST", digits)
    state = sm.get("TEST")
    bundle = build_features(state, windows=[500])

    model = MarkovModel(max_order=3)
    probs = model.predict(state, bundle)
    assert abs(probs.sum() - 1.0) < 1e-6
    # with a rare/unsupported high-order context, the model should back off
    # to something much closer to uniform rather than a spike
    assert probs.max() < 0.5
