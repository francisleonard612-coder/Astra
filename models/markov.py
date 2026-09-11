from __future__ import annotations

import numpy as np

from features.feature_engine import FeatureBundle
from models.base import DigitModel, N_DIGITS, normalize
from state.rolling_state import SymbolState

MIN_CONTEXT_COUNT = 20  # a transition context needs at least this many observations
                        # before Astra trusts it over the next-lower order


class MarkovModel(DigitModel):
    """P(next digit | previous k digits), k up to max_order, with backoff to a
    lower order (and ultimately to the digit's own unconditional frequency)
    whenever the higher-order context hasn't been seen often enough. This is
    the guard against "higher-order models when sample sizes become
    inadequate" (spec section 8).

    Reads counts directly from `state.markov_row()`, which is maintained
    incrementally in O(max_order) per tick (see state/rolling_state.py).
    This used to rescan the entire digit history from scratch on every
    single predict() call -- an O(window) cost per order, repeated for the
    "is this context well-supported" check too -- which became the dominant
    per-tick cost once a symbol's history grew into the thousands. Fixed
    after profiling showed it turning a 500-tick backtest into a multi-
    second run.
    """
    name = "markov"

    def __init__(self, max_order: int = 3):
        self.max_order = max_order

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        seq = state.digits
        n = len(seq)
        for order in range(self.max_order, 0, -1):
            if n < order:
                continue
            # direct deque indexing near the right end -- O(order), not
            # list(seq)[-order:] which would copy the whole history first
            context = tuple(seq[n - order + i] for i in range(order))
            row, support = state.markov_row(order, context)
            if row is not None and support > 0 and (support >= MIN_CONTEXT_COUNT or order == 1):
                return normalize(row)
        # ultimate fallback: unconditional frequency over the full history
        stats = bundle.windows.get(max(bundle.windows.keys())) if bundle.windows else None
        if stats:
            return normalize(np.array(stats["freq"]))
        return np.full(N_DIGITS, 1.0 / N_DIGITS)

    def is_ready(self, state: SymbolState) -> bool:
        return state.total_observed >= 50
