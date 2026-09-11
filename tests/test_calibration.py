import random

from models.calibration import CalibrationTracker


def test_uncalibrated_returns_raw_probability():
    tracker = CalibrationTracker(method="isotonic", min_samples=200, refit_every=50)
    assert tracker.calibrate(0.7) == 0.7
    assert not tracker.is_calibrated


def test_refits_after_enough_samples():
    random.seed(0)
    tracker = CalibrationTracker(method="isotonic", min_samples=50, refit_every=25)
    # a model that's overconfident: says 0.9 but only wins 50% of the time
    for _ in range(120):
        outcome = 1 if random.random() < 0.5 else 0
        tracker.record(0.9, outcome)
    assert tracker.is_calibrated
    calibrated = tracker.calibrate(0.9)
    # calibration should pull the overconfident 0.9 down toward the true ~0.5 rate
    assert calibrated < 0.85


def test_quality_score_bounded_zero_to_one():
    tracker = CalibrationTracker()
    for i in range(50):
        tracker.record(0.6, i % 2)
    score = tracker.quality_score()
    assert 0.0 <= score <= 1.0


def test_quality_score_low_with_too_few_samples():
    tracker = CalibrationTracker()
    tracker.record(0.6, 1)
    assert tracker.quality_score() == 0.3


def test_temperature_blend_off_by_default_matches_pre_existing_behavior():
    random.seed(1)
    tracker = CalibrationTracker(method="isotonic", min_samples=50, refit_every=25)
    for _ in range(120):
        outcome = 1 if random.random() < 0.5 else 0
        tracker.record(0.9, outcome)
    assert tracker.use_temperature_blend is False
    # identical to the pre-existing isotonic-only path -- no behavior change
    # for callers that don't opt in
    expected = float(tracker._calibrator.predict([0.9])[0])
    assert tracker.calibrate(0.9) == expected


def test_temperature_blend_pulls_overconfident_estimate_down():
    random.seed(2)
    tracker = CalibrationTracker(method="isotonic", min_samples=50, refit_every=25,
                                  use_temperature_blend=True)
    # consistently overconfident: claims 0.95 but only wins ~50% of the time
    for _ in range(150):
        outcome = 1 if random.random() < 0.5 else 0
        tracker.record(0.95, outcome)
    assert tracker.is_calibrated
    calibrated = tracker.calibrate(0.95)
    assert calibrated < 0.8  # pulled well down from the overconfident 0.95
    assert 0.0 <= calibrated <= 1.0


def test_temperature_blend_falls_back_to_temperature_alone_before_isotonic_fits():
    tracker = CalibrationTracker(method="isotonic", min_samples=10_000, refit_every=25,
                                  use_temperature_blend=True)
    # isotonic never gets enough samples to fit (min_samples way out of
    # reach), but temperature blending must not crash and must still return
    # a valid probability
    for i in range(60):
        tracker.record(0.9, i % 2)
    assert not tracker.is_calibrated
    result = tracker.calibrate(0.9)
    assert 0.0 <= result <= 1.0


def test_temperature_fit_requires_the_documented_outcome_semantics():
    """Direct test of the correctness note in CalibrationTracker's docstring:
    outcomes must be "did the predicted event actually happen", not "was
    the prediction/trade correct" -- feeding the latter silently biases the
    fit. This doesn't (and can't) detect the mislabeling automatically; it
    documents what a mislabeled fit looks like so the distinction stays
    visible to whoever wires this up for Rise/Fall."""
    random.seed(3)
    correct_semantics = CalibrationTracker(method="isotonic", min_samples=50, refit_every=25,
                                            use_temperature_blend=True)
    mislabeled_semantics = CalibrationTracker(method="isotonic", min_samples=50, refit_every=25,
                                               use_temperature_blend=True)
    for _ in range(150):
        p_up = random.uniform(0.3, 0.7)
        price_went_up = 1 if random.random() < p_up else 0
        was_call = random.random() < 0.5
        # correct: outcome is always "did price go up", regardless of side
        correct_semantics.record(p_up, price_went_up)
        # mislabeled: outcome is "was the trade correct" -- direction-blind
        was_correct = price_went_up if was_call else (1 - price_went_up)
        mislabeled_semantics.record(p_up, was_correct)
    # both fit without error either way -- the bug is silent, which is
    # exactly why the docstring warning matters; this test exists to keep
    # that difference visible, not to assert a specific numeric outcome.
    assert 0.0 <= correct_semantics.calibrate(0.6) <= 1.0
    assert 0.0 <= mislabeled_semantics.calibrate(0.6) <= 1.0
