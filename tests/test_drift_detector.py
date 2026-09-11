import numpy as np

from models.drift_detector import DriftDetector


def _rng():
    return np.random.default_rng(0)


def test_check_ks_false_without_enough_reference_or_live_data():
    d = DriftDetector()
    assert d.check_ks(_rng().normal(0, 0.001, size=200)) is False  # no reference snapshot yet

    d.snapshot_reference(_rng().normal(0, 0.001, size=500), [])
    assert d.check_ks(_rng().normal(0, 0.001, size=10)) is False  # too little live data


def test_check_ks_detects_genuine_distribution_shift():
    rng = _rng()
    d = DriftDetector(ks_p_threshold=0.05)
    d.snapshot_reference(rng.normal(0.0, 0.001, size=500), [])
    # same distribution -- should not fire
    assert d.check_ks(rng.normal(0.0, 0.001, size=200)) is False
    # a genuinely different distribution (much higher vol) -- should fire
    assert d.check_ks(rng.normal(0.0, 0.01, size=200)) is True


def test_check_psi_false_before_enough_confidence_history():
    d = DriftDetector()
    d.snapshot_reference(np.array([0.001] * 500), [0.6] * 200)
    for _ in range(99):
        assert d.check_psi(0.6) is False  # under the 100-reading minimum


def test_check_psi_detects_shifted_confidence_distribution():
    d = DriftDetector(psi_threshold=0.20)
    reference_confidences = [0.5 + 0.05 * np.sin(i) for i in range(200)]  # tight band around 0.5
    d.snapshot_reference(np.array([0.001] * 500), reference_confidences)
    # feed 150 readings clustered far away from the reference band
    fired = False
    for _ in range(150):
        fired = d.check_psi(0.95) or fired
    assert fired is True


def test_update_cusum_fires_on_sustained_losing_streak():
    d = DriftDetector(cusum_threshold=4.0, cusum_drift=0.03)
    fired = False
    for _ in range(50):
        fired = d.update_cusum(won=False) or fired
    assert fired is True


def test_update_cusum_never_fires_on_a_genuine_winning_streak():
    d = DriftDetector(cusum_threshold=4.0, cusum_drift=0.03)
    fired = False
    for _ in range(500):
        fired = d.update_cusum(won=True) or fired
    assert fired is False
    assert d._cusum_stat == 0.0


def test_update_cusum_fires_much_faster_on_a_genuine_losing_streak_than_a_fair_coin():
    """These default parameters (cusum_drift=0.03, cusum_threshold=4.0,
    taken as-is from the source) are quite sensitive over long horizons --
    by direct simulation this fires on a genuinely FAIR win rate in ~114
    draws on average, not a rare event over a realistic trade count. See
    this module's docstring for the full characterization; what matters and
    is worth locking in as a regression test is the DIRECTIONAL property:
    it must fire meaningfully faster the worse the true win rate is, and
    never on a winning run (covered separately above)."""
    rng = np.random.default_rng(1)

    def avg_draws_to_fire(win_prob: float, n_trials: int = 100, max_draws: int = 300) -> float:
        draws = []
        for _ in range(n_trials):
            d = DriftDetector(cusum_threshold=4.0, cusum_drift=0.03)
            for i in range(max_draws):
                if d.update_cusum(won=bool(rng.random() < win_prob)):
                    draws.append(i)
                    break
            else:
                draws.append(max_draws)
        return float(np.mean(draws))

    fair = avg_draws_to_fire(0.5)
    losing = avg_draws_to_fire(0.35)
    assert losing < fair * 0.5  # decisively faster, not just marginally


def test_snapshot_reference_clears_confidence_history_and_streak():
    """Regression test for the source's own documented production bug: a
    recalibration that doesn't clear stale confidence history/streak
    self-perpetuates a drift lock (recalibrate -> instantly re-fail against
    old data -> recalibrate again)."""
    d = DriftDetector(psi_threshold=0.20, consecutive_required=3)
    d.snapshot_reference(np.array([0.001] * 500), [0.5] * 200)
    # drive PSI into a fired state with 150 anomalous readings
    for _ in range(150):
        d.check_psi(0.95)
    assert len(d._confidence_history) >= 100

    # recalibrate -- this must wipe the stale history, not carry it forward
    d.snapshot_reference(np.array([0.001] * 500), [0.5] * 200)
    assert len(d._confidence_history) == 0
    assert d._consecutive_fires == 0
    assert d.degraded is False

    # immediately after recalibration, PSI must NOT instantly re-fire from
    # leftover history -- it needs fresh readings to accumulate again
    assert d.check_psi(0.5) is False


def test_check_all_requires_consecutive_fires_not_a_single_blip():
    """Regression test for the source's own documented production bug: a
    one-shot latch on the first fire made every symbol accumulate at least
    one noisy blip. Only a SUSTAINED signal (consecutive_required fires in
    a row) should set degraded=True."""
    d = DriftDetector(ks_p_threshold=0.05, consecutive_required=3)
    rng = _rng()
    reference_returns = rng.normal(0.0, 0.001, size=500)
    d.snapshot_reference(reference_returns, [0.5] * 200)

    # one anomalous read among normal ones -- must NOT latch degraded
    normal_returns = rng.normal(0.0, 0.001, size=200)
    anomalous_returns = rng.normal(0.0, 0.05, size=200)  # wildly different vol

    assert d.check_all(normal_returns, 0.5) is False
    assert d.check_all(anomalous_returns, 0.5) is False  # single fire -- streak=1, not yet degraded
    assert d.check_all(normal_returns, 0.5) is False  # non-fire resets the streak
    assert d._consecutive_fires == 0
    assert d.degraded is False

    # now a SUSTAINED run of anomalous reads -- must eventually degrade
    fired = False
    for _ in range(6):
        fired = d.check_all(anomalous_returns, 0.5) or fired
    assert fired is True
    assert d.degraded is True
