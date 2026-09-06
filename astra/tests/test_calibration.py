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
