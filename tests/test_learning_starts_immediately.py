import numpy as np

from app.config import get_config
from decision.decision_engine import SymbolPipeline
from state.rolling_state import StateManager


def test_markov_counts_accumulate_from_the_first_tick():
    sm = StateManager(max_window=100, max_markov_order=2)
    state = sm.get("X")
    state.push(3)
    state.push(7)
    # order-1 context (3,) -> 7 should already be recorded after just two ticks
    row, support = state.markov_row(1, (3,))
    assert row is not None
    assert support == 1
    assert row[7] == 1


def test_online_models_learn_before_the_trading_sample_size_threshold():
    """min_samples_per_symbol (default 300) gates TRADING decisions, not
    learning. Models, the per-digit performance tracker, and calibration
    should all already be accumulating well before that -- there's no reason
    to throw away the first ~300 ticks of learning signal just because
    Astra isn't ready to risk money yet."""
    cfg = get_config()
    sm = StateManager(max_window=cfg.get("feature_windows", default=[2500])[-1],
                       max_markov_order=cfg.get("max_markov_order", default=3))
    state = sm.get("X")
    pipeline = SymbolPipeline("X", cfg)

    rng = np.random.default_rng(0)
    pending = None
    # well under the 300-sample trading threshold
    for i in range(120):
        digit = int(rng.integers(0, 10))
        if pending is not None:
            predictions, bundle = pending
            pipeline.observe(state, bundle, predictions, digit, over_barrier=2, under_barrier=7)
        state.push(digit)
        predictions, bundle = pipeline.predict(state)
        pending = (predictions, bundle)

    # the logistic model should have received partial_fit calls and be fitted
    assert pipeline.registry.models["logistic"]._fitted
    # the performance tracker should have live per-digit log-loss data, not
    # just the config defaults
    assert pipeline.performance.rolling_log_loss("markov") is not None
    # batch models (Random Forest / XGBoost) are buffering samples even
    # though they haven't hit their (much higher) min_observations yet
    rf = pipeline.registry.models["random_forest"]
    assert len(rf._X) == 119  # one fewer than ticks -- the very first tick has no prior "pending" to learn from
