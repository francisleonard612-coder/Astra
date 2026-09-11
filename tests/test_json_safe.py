import json

import numpy as np

from database.repository import _json_safe


def test_numpy_bool_is_converted_and_serializable():
    # This is the exact bug caught from a live deployment log: comparing two
    # numpy.float64 values (as happens throughout champion_challenger.py and
    # architecture_competition.py) produces numpy.bool, which -- unlike
    # numpy.float64 -- does NOT subclass Python's bool (bool can't be
    # subclassed), so it silently failed json.dumps every time.
    numpy_bool = np.float64(2.0) > np.float64(1.0)
    assert type(numpy_bool).__name__ in ("bool_", "bool")  # numpy's type, not Python's
    safe = _json_safe({"stable": numpy_bool})
    assert isinstance(safe["stable"], bool)
    json.dumps(safe)  # must not raise


def test_numpy_float_and_int_are_converted():
    safe = _json_safe({"loss": np.float64(0.5), "count": np.int64(3)})
    assert isinstance(safe["loss"], float)
    assert isinstance(safe["count"], int)
    json.dumps(safe)


def test_numpy_array_is_converted_to_list():
    safe = _json_safe({"weights": np.array([0.1, 0.2, 0.3])})
    assert isinstance(safe["weights"], list)
    assert all(isinstance(v, float) for v in safe["weights"])
    json.dumps(safe)


def test_nested_dicts_and_lists_are_sanitized():
    payload = {
        "metrics": {
            "scores": {"global": np.float64(0.7), "specialist": np.float64(0.6)},
            "flags": [np.bool_(True), np.bool_(False), True, False],
        }
    }
    safe = _json_safe(payload)
    json.dumps(safe)  # must not raise
    assert safe["metrics"]["flags"] == [True, False, True, False]


def test_plain_python_values_pass_through_unchanged():
    payload = {"a": 1, "b": "text", "c": None, "d": [1, 2, 3], "e": True}
    assert _json_safe(payload) == payload
