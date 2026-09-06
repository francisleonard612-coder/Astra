"""
Per-symbol online state: a bounded history of observed digits, streak/gap
counters, and incrementally-maintained Markov transition counts -- all
updated in O(max_markov_order) per tick rather than by rescanning history.
Everything else (frequency, entropy) is derived on demand from the digit
deque, which is cheap since it's bounded to `max_window`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SymbolState:
    symbol: str
    max_window: int
    max_markov_order: int = 3
    digits: deque = field(default_factory=deque)
    last_digit: int | None = None
    same_digit_streak: int = 0
    high_streak: int = 0   # digit >= 5
    low_streak: int = 0    # digit <= 4
    over2_streak: int = 0  # digit > 2
    under7_streak: int = 0  # digit < 7
    last_seen_tick_index: dict = field(default_factory=dict)  # digit -> tick index of last occurrence
    tick_index: int = 0
    total_observed: int = 0
    # order -> {context_tuple: length-10 raw count array}. Maintained
    # incrementally (O(max_markov_order) per push -- see push()) instead of
    # rescanning the whole digit history every tick, which used to be the
    # single most expensive thing Astra did per tick once history grew into
    # the thousands (an O(n) full-history rescan, once per order, every
    # single tick -- see the git history / PR notes for the profiling that
    # caught this). Support count for a context is just that context's
    # count array sum, so this also makes the Markov model's "is this
    # context well-supported" check free instead of another O(n) scan.
    markov_counts: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.digits, deque):
            self.digits = deque(maxlen=self.max_window)
        else:
            self.digits = deque(self.digits, maxlen=self.max_window)
        if not self.markov_counts:
            self.markov_counts = {o: {} for o in range(1, self.max_markov_order + 1)}

    def push(self, digit: int) -> None:
        digits_full = len(self.digits) == self.digits.maxlen
        n = len(self.digits)

        for order, table in self.markov_counts.items():
            # the digit about to fall out of the window (if any) invalidates
            # exactly one transition per order: the one whose context started
            # at the very oldest retained digit. Indexed directly on the
            # deque (O(order) near either end) -- NOT via list(self.digits),
            # which would silently reintroduce an O(window) full-history
            # copy on every single tick and defeat the point of this method.
            if digits_full and n > order:
                old_context = tuple(self.digits[i] for i in range(order))
                old_next = self.digits[order]
                row = table.get(old_context)
                if row is not None:
                    row[old_next] = max(0.0, row[old_next] - 1)
                    if not row.any():
                        del table[old_context]

            if n >= order:
                new_context = tuple(self.digits[n - order + i] for i in range(order))
                row = table.get(new_context)
                if row is None:
                    row = np.zeros(10)
                    table[new_context] = row
                row[digit] += 1

        self.digits.append(digit)
        self.tick_index += 1
        self.total_observed += 1

        if self.last_digit is not None and digit == self.last_digit:
            self.same_digit_streak += 1
        else:
            self.same_digit_streak = 1

        is_high = digit >= 5
        self.high_streak = self.high_streak + 1 if is_high else 0
        self.low_streak = self.low_streak + 1 if not is_high else 0

        is_over2 = digit > 2
        self.over2_streak = self.over2_streak + 1 if is_over2 else 0
        is_under7 = digit < 7
        self.under7_streak = self.under7_streak + 1 if is_under7 else 0

        self.last_seen_tick_index[digit] = self.tick_index
        self.last_digit = digit

    def markov_row(self, order: int, context: tuple[int, ...]) -> tuple[np.ndarray | None, int]:
        """Returns (raw_count_array, support_count) for a context at a given
        order, or (None, 0) if that context hasn't been seen. O(1)."""
        row = self.markov_counts.get(order, {}).get(context)
        if row is None:
            return None, 0
        return row, int(row.sum())

    def gap(self, digit: int) -> int:
        """Ticks since `digit` last appeared (0 if it's the current tick)."""
        last = self.last_seen_tick_index.get(digit)
        if last is None:
            return self.tick_index  # never seen -- gap is "everything so far"
        return self.tick_index - last

    def window(self, size: int) -> list[int]:
        size = min(size, len(self.digits))
        if size <= 0:
            return []
        return list(self.digits)[-size:]

    def has_min_samples(self, minimum: int) -> bool:
        return self.total_observed >= minimum


class StateManager:
    def __init__(self, max_window: int, max_markov_order: int = 3):
        self.max_window = max_window
        self.max_markov_order = max_markov_order
        self._states: dict[str, SymbolState] = {}

    def get(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState(
                symbol=symbol, max_window=self.max_window, max_markov_order=self.max_markov_order,
            )
        return self._states[symbol]

    def seed(self, symbol: str, digits: list[int]) -> None:
        state = self.get(symbol)
        for d in digits:
            state.push(d)
