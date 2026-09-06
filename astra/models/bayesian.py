from __future__ import annotations

import numpy as np

from features.feature_engine import FeatureBundle
from models.base import DigitModel, N_DIGITS, normalize
from state.rolling_state import SymbolState


class BayesianModel(DigitModel):
    """Dirichlet-multinomial posterior mean over digit frequency.

    Starts at a uniform prior (alpha_0 per digit) and updates with observed
    counts within the model's own window. Behaves like rolling frequency for
    large samples but is better behaved (no divide-by-zero, sensible under
    very little data) early on -- which is exactly when Astra needs to start
    producing usable estimates per section 1 (online learning from minute one).
    """
    name = "bayesian"

    def __init__(self, prior_strength: float = 5.0, window: int = 1000):
        self.prior_strength = prior_strength
        self.window = window

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        alpha0 = self.prior_strength / N_DIGITS
        stats = bundle.windows.get(self.window)
        if stats is None:
            counts = np.zeros(N_DIGITS)
        else:
            counts = np.array(stats["counts"])
        posterior = counts + alpha0
        return normalize(posterior)

    def is_ready(self, state: SymbolState) -> bool:
        return state.total_observed >= 30
