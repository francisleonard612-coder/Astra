import numpy as np

from pricing.monte_carlo_duration import hmm_gbm_terminal_log_returns, monte_carlo_duration


def test_too_little_history_falls_back_to_first_candidate():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.001, size=5)
    dur, p = monte_carlo_duration(returns, direction=1, candidate_durations=[5, 10, 20], rng=rng)
    assert dur == 5
    assert p == 0.5


def test_returns_a_candidate_from_the_input_list():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.001, size=200)
    candidates = [3, 7, 15]
    dur, p = monte_carlo_duration(returns, direction=1, candidate_durations=candidates, n_sims=500, rng=rng)
    assert dur in candidates
    assert 0.0 <= p <= 1.0


def test_strong_genuine_drift_is_detected():
    # Not pure noise this time -- a real, strong, sustained upward drift.
    # The estimator should correctly favor Rise with high confidence.
    rng = np.random.default_rng(2)
    returns = rng.normal(0.01, 0.001, size=200)  # drift is 10x the noise scale
    dur, p = monte_carlo_duration(returns, direction=1, candidate_durations=[5, 10, 20], n_sims=3000, rng=rng)
    assert p > 0.8


def test_empirical_win_rate_blend_pulls_estimate_toward_empirical():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.001, size=200)
    _, p_plain = monte_carlo_duration(returns, direction=1, candidate_durations=[10], n_sims=2000, rng=rng)
    _, p_blended = monte_carlo_duration(
        returns, direction=1, candidate_durations=[10], n_sims=2000, rng=rng,
        empirical_win_rates={10: 0.95},
    )
    # blend is 70% empirical / 30% simulation -- a very high empirical rate
    # should pull the blended estimate up substantially versus the
    # unblended (simulation-only) estimate on pure noise (~0.5)
    assert p_blended > p_plain
    assert p_blended > 0.6


def test_hmm_gbm_terminal_log_returns_falls_back_without_a_model():
    rng = np.random.default_rng(4)
    recent = rng.normal(0.0, 0.001, size=50)
    out = hmm_gbm_terminal_log_returns(None, recent, fallback_vol=0.001, n_steps=10, n_sims=1000, rng=rng)
    assert len(out) == 1000
    assert abs(float(np.mean(out))) < 0.01  # flat Gaussian, no drift injected


def test_direction_produces_complementary_not_invariant_probabilities():
    """Regression test for a bug found while testing this port (not present
    in the source, which never exercises this case -- see the module
    docstring's "PORT-TIME FIX"): CALL and PUT win-probability estimates for
    the SAME returns must respond oppositely to genuine drift, not collapse
    to the same value regardless of direction."""
    rng = np.random.default_rng(9)
    returns = rng.normal(-0.02, 0.001, size=200)  # strong, unambiguous bearish drift

    _, p_call = monte_carlo_duration(returns, direction=1, candidate_durations=[10],
                                      n_sims=5000, rng=np.random.default_rng(1))
    _, p_put = monte_carlo_duration(returns, direction=-1, candidate_durations=[10],
                                     n_sims=5000, rng=np.random.default_rng(1))

    assert p_put > 0.9   # PUT should win decisively on real bearish drift
    assert p_call < 0.1  # CALL should lose decisively on the same data
    assert abs((p_call + p_put) - 1.0) < 0.05  # complementary, not both collapsing to 0


def test_v10_fix_no_duration_bias_under_pure_noise():
    """Direct regression test for the exact bug documented in the source
    repo's README ("v10: minutes only -- two real MC bugs fixed"): under
    pure noise (zero true drift), the estimated win probability at each
    candidate duration must not systematically climb above 0.5 as duration
    grows. The pre-fix version (direction * abs(mean) instead of the signed
    mean, no drift-SE-in-quadrature term) does exactly that -- see this
    module's own docstring and pricing/monte_carlo_duration.py's inline
    comments for the full mechanism."""
    rng = np.random.default_rng(42)
    vol = 0.001
    durations = [1, 10, 30, 60]
    n_trials = 150

    mean_p_by_duration = {}
    for d in durations:
        probs = []
        for _ in range(n_trials):
            returns = rng.normal(0.0, vol, size=200)
            _, p = monte_carlo_duration(returns, direction=1, candidate_durations=[d], n_sims=800, rng=rng)
            probs.append(p)
        mean_p_by_duration[d] = float(np.mean(probs))

    for d, mean_p in mean_p_by_duration.items():
        assert 0.40 < mean_p < 0.60, f"duration {d} averaged {mean_p:.3f} across {n_trials} pure-noise trials"

    # the real signature of the bug: a systematic INCREASE from short to
    # long duration. Assert the longest candidate isn't meaningfully higher
    # than the shortest -- the fixed version should be flat (if anything,
    # very slightly conservative at longer durations), not climbing.
    assert mean_p_by_duration[durations[-1]] < mean_p_by_duration[durations[0]] + 0.08
