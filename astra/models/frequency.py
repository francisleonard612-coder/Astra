from __future__ import annotations

import numpy as np

from features.feature_engine import FeatureBundle
from models.base import DigitModel, N_DIGITS, normalize
from state.rolling_state import SymbolState


class UniformModel(DigitModel):
    """The null hypothesis: every digit equally likely. The mandatory sanity baseline
    every other model must beat to justify its ensemble weight."""
    name = "uniform"

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        return np.full(N_DIGITS, 1.0 / N_DIGITS)


class RollingFrequencyModel(DigitModel):
    """Empirical frequency over a configured window."""
    name = "rolling_frequency"

    def __init__(self, window: int = 500):
        self.window = window

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        stats = bundle.windows.get(self.window)
        if stats is None or stats["n"] == 0:
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        return normalize(np.array(stats["freq"]))

    def is_ready(self, state: SymbolState) -> bool:
        return state.total_observed >= max(30, self.window // 10)


class EWMAFrequencyModel(DigitModel):
    """Exponentially weighted frequency -- reacts faster to recent regime changes
    than a fixed rolling window, at the cost of more noise."""
    name = "ewma_frequency"

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self._ewma: dict[str, np.ndarray] = {}

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        vec = self._ewma.get(state.symbol)
        if vec is None:
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        return normalize(vec)

    def observe(self, state: SymbolState, bundle: FeatureBundle, actual_digit: int) -> None:
        onehot = np.zeros(N_DIGITS)
        onehot[actual_digit] = 1.0
        prev = self._ewma.get(state.symbol)
        if prev is None:
            self._ewma[state.symbol] = onehot
        else:
            self._ewma[state.symbol] = (1 - self.alpha) * prev + self.alpha * onehot

    def is_ready(self, state: SymbolState) -> bool:
        return state.symbol in self._ewma and state.total_observed >= 50
