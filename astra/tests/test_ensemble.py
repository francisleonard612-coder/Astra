import numpy as np

from models.ensemble import combine, model_agreement


def test_combine_weighted_average_sums_to_one():
    preds = {
        "a": np.array([0.1] * 10),
        "b": np.array([0.05] * 5 + [0.15] * 5),
    }
    weights = {"a": 0.5, "b": 0.5}
    result = combine(preds, weights)
    assert abs(result.sum() - 1.0) < 1e-9


def test_combine_ignores_zero_weight_models():
    preds = {
        "a": np.full(10, 0.1),
        "b": np.array([1.0] + [0.0] * 9),  # would badly skew result if included
    }
    weights = {"a": 1.0, "b": 0.0}
    result = combine(preds, weights)
    assert np.allclose(result, np.full(10, 0.1), atol=1e-6)


def test_model_agreement_high_when_models_agree():
    preds = {
        "a": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
        "b": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
    }
    agreement = model_agreement(preds, over_barrier=2, under_barrier=7)
    assert agreement["over_agreement"] > 0.95
    assert agreement["under_agreement"] > 0.95


def test_model_agreement_low_when_models_disagree():
    preds = {
        "a": np.array([0.0] * 9 + [1.0]),   # thinks digit is always 9 -> high P(over2)
        "b": np.array([1.0] + [0.0] * 9),   # thinks digit is always 0 -> low P(over2)
    }
    agreement = model_agreement(preds, over_barrier=2, under_barrier=7)
    assert agreement["over_agreement"] < 0.5
