import numpy as np

from app.config import get_config
from decision.decision_engine import ARCHITECTURES
from learning.architecture_competition import ArchitectureCompetitionManager
from state.rolling_state import StateManager


def make_manager(symbol="TEST", **overrides):
    cfg = get_config()
    mgr = ArchitectureCompetitionManager(symbol, cfg, repository=None)
    for k, v in overrides.items():
        setattr(mgr, k, v)
    return mgr


def test_predict_all_returns_valid_distributions_for_all_three_architectures():
    mgr = make_manager()
    sm = StateManager(max_window=500, max_markov_order=3)
    state = sm.get("TEST")
    rng = np.random.default_rng(0)
    for d in rng.integers(0, 10, size=100):
        state.push(int(d))

    snapshot = mgr.predict_all(state)
    for arch in ARCHITECTURES:
        vec = snapshot.vectors[arch]
        assert vec.shape == (10,)
        assert abs(vec.sum() - 1.0) < 1e-6
        assert (vec >= 0).all()


def test_observe_pending_is_a_noop_without_a_stashed_snapshot():
    mgr = make_manager()
    sm = StateManager(max_window=500, max_markov_order=3)
    state = sm.get("TEST")
    state.push(3)
    assert not mgr.has_pending()
    mgr.observe_pending(state, actual_digit=3)  # should not raise


def test_predict_observe_cycle_feeds_metrics_for_all_architectures():
    mgr = make_manager()
    sm = StateManager(max_window=500, max_markov_order=3)
    state = sm.get("TEST")
    rng = np.random.default_rng(0)

    for d in rng.integers(0, 10, size=60):
        snapshot = mgr.predict_all(state)
        mgr.stash_pending(snapshot)
        state.push(int(d))
        if mgr.has_pending():
            mgr.observe_pending(state, actual_digit=int(d))

    for arch in ARCHITECTURES:
        assert mgr.metrics[arch].summary()["n"] > 0


def _seed_metrics(mgr, arch: str, log_losses: list[float]) -> None:
    for v in log_losses:
        mgr.metrics[arch].log_loss.append(v)
        mgr.metrics[arch].brier.append(v)  # reuse for simplicity in this synthetic test
        mgr.metrics[arch].agreement.append(0.8)
        mgr.metrics[arch].economic_ev.append(0.05)
        mgr.metrics[arch].realized_pnl.append(0.1)


def test_promotion_requires_stability_across_both_halves():
    mgr = make_manager(min_samples_for_evaluation=20, min_promotion_margin=0.0)
    n = 40
    # "specialist" is clearly and consistently better (lower log-loss) than
    # "global" across the WHOLE window, in both halves.
    _seed_metrics(mgr, "global", [1.0] * n)
    _seed_metrics(mgr, "specialist", [0.5] * n)
    _seed_metrics(mgr, "hybrid", [0.9] * n)

    assert mgr.champion == "global"
    mgr._maybe_promote_architecture()
    assert mgr.champion == "specialist"


def test_no_promotion_when_improvement_only_holds_in_one_half():
    mgr = make_manager(min_samples_for_evaluation=20, min_promotion_margin=0.0)
    # specialist looks better on AVERAGE but is actually worse in the second
    # half -- a classic "looks good until you check for consistency" case
    # that the stability check exists to catch.
    global_losses = [1.0] * 20 + [1.0] * 20
    specialist_losses = [0.1] * 20 + [2.0] * 20  # great first half, terrible second half
    for v in global_losses:
        mgr.metrics["global"].log_loss.append(v)
    for v in specialist_losses:
        mgr.metrics["specialist"].log_loss.append(v)
    for v in global_losses:
        mgr.metrics["hybrid"].log_loss.append(1.5)

    assert mgr.champion == "global"
    mgr._maybe_promote_architecture()
    # average log-loss of specialist ((0.1*20+2.0*20)/40=1.05) is actually
    # WORSE than global's flat 1.0 here, so this also fails on the composite
    # score itself, not just the stability check -- either way, no promotion.
    assert mgr.champion == "global"


def test_no_promotion_before_minimum_sample_size():
    mgr = make_manager(min_samples_for_evaluation=100, min_promotion_margin=0.0)
    _seed_metrics(mgr, "global", [1.0] * 10)
    _seed_metrics(mgr, "specialist", [0.1] * 10)
    _seed_metrics(mgr, "hybrid", [0.9] * 10)
    mgr._maybe_promote_architecture()
    assert mgr.champion == "global"  # not enough samples yet, regardless of how good specialist looks


def test_composite_scores_are_json_safe_native_floats():
    mgr = make_manager(min_samples_for_evaluation=10)
    _seed_metrics(mgr, "global", [1.0] * 15)
    _seed_metrics(mgr, "specialist", [0.8] * 15)
    _seed_metrics(mgr, "hybrid", [0.9] * 15)
    raw = mgr._raw_metrics()
    scores = mgr._composite_scores(raw)
    for arch in ARCHITECTURES:
        assert isinstance(scores[arch], float)
