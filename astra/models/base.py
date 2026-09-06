"""
Common interface every digit-probability model implements.

`predict` must be side-effect-free and fast (called on every tick, per
symbol). `observe` is where a model gets to learn from the realized outcome
-- cheap/incremental models update immediately; batch models (Random
Forest, XGBoost) just buffer the observation and retrain on a cadence
controlled by learning/retraining.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from features.feature_engine import FeatureBundle
from state.rolling_state import SymbolState

N_DIGITS = 10


class DigitModel(ABC):
    name: str
    #  active | inactive | experimental | champion | challenger | deprecated
    status: str = "active"

    @abstractmethod
    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        """Return a normalized length-10 probability vector."""
        raise NotImplementedError

    def observe(self, state: SymbolState, bundle: FeatureBundle, actual_digit: int) -> None:
        """Optional: incorporate the realized outcome. No-op by default."""
        return

    def is_ready(self, state: SymbolState) -> bool:
        return True


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.clip(vec, 1e-9, None)
    return vec / vec.sum()
