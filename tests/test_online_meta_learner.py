import numpy as np

from models.online_meta_learner import OnlineMetaLearner


def test_predict_none_below_min_samples():
    m = OnlineMetaLearner(n_features=3, min_samples=50)
    for _ in range(20):
        m.update([0.1, 0.2, 0.3], 1.0)
    assert m.predict([0.1, 0.2, 0.3]) is None
    assert m.is_ready is False


def test_predict_available_once_min_samples_reached():
    m = OnlineMetaLearner(n_features=3, min_samples=20)
    for _ in range(25):
        m.update([0.1, 0.2, 0.3], 1.0)
    p = m.predict([0.1, 0.2, 0.3])
    assert p is not None
    assert 0.0 <= p <= 1.0
    assert m.is_ready is True


def test_online_update_learns_a_simple_linearly_separable_pattern():
    rng = np.random.default_rng(0)
    m = OnlineMetaLearner(n_features=2, min_samples=30, learning_rate=0.1)
    # y = 1 when x[0] > 0, else 0 -- simple, clearly learnable signal
    for _ in range(2000):
        x = rng.normal(0.0, 1.0, size=2)
        y = 1.0 if x[0] > 0 else 0.0
        m.update(x, y)

    # test on fresh, clearly-separated points
    assert m.predict([3.0, 0.0]) > 0.8
    assert m.predict([-3.0, 0.0]) < 0.2


def test_retrain_from_buffer_converges_from_a_batch():
    rng = np.random.default_rng(1)
    m = OnlineMetaLearner(n_features=2, min_samples=30, learning_rate=0.1)
    xs, ys = [], []
    for _ in range(500):
        x = rng.normal(0.0, 1.0, size=2)
        y = 1.0 if x[0] > 0 else 0.0
        xs.append(x)
        ys.append(y)
        m._buffer.append((np.array(x), y))  # seed the buffer directly, bypassing incremental update()

    m.retrain_from_buffer(epochs=100)
    assert m.predict([3.0, 0.0]) > 0.8
    assert m.predict([-3.0, 0.0]) < 0.2


def test_retrain_from_buffer_is_a_noop_below_min_samples():
    m = OnlineMetaLearner(n_features=2, min_samples=100)
    for _ in range(10):
        m.update([1.0, 1.0], 1.0)
    m.retrain_from_buffer()  # must not raise despite too few samples
    assert m.predict([1.0, 1.0]) is None
