"""
Rolling price/return history for Rise/Fall, at both native tick resolution
and aggregated 1-minute bars.

The digit bot never needed this -- state/rolling_state.py's SymbolState
tracks only last-digits, never the raw quote. pricing/monte_carlo_duration.py
needs actual log-returns, separately at BOTH tick resolution (for tick-
duration candidates) and minute-bar resolution (for minute-duration
candidates) -- see that module's docstring for why ticks and minutes must
never be mixed into a single Monte Carlo call.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PriceSeries:
    symbol: str
    max_tick_window: int = 2500
    minute_bar_seconds: int = 60
    max_minute_window: int = 1000

    prices: deque = field(default_factory=deque)           # raw quotes, tick resolution
    tick_log_returns: deque = field(default_factory=deque)
    minute_log_returns: deque = field(default_factory=deque)

    _minute_bucket_start: int | None = field(default=None, repr=False)
    _minute_bucket_last_price: float | None = field(default=None, repr=False)
    _last_minute_close: float | None = field(default=None, repr=False)

    def __post_init__(self):
        self.prices = deque(self.prices, maxlen=self.max_tick_window)
        self.tick_log_returns = deque(self.tick_log_returns, maxlen=self.max_tick_window)
        self.minute_log_returns = deque(self.minute_log_returns, maxlen=self.max_minute_window)

    def push(self, epoch: int, price: float) -> None:
        if price <= 0:
            return  # log-return undefined for a non-positive price -- skip rather than poison the series
        if self.prices:
            prev = self.prices[-1]
            if prev > 0:
                self.tick_log_returns.append(math.log(price / prev))
        self.prices.append(price)
        self._push_minute_bar(epoch, price)

    def _push_minute_bar(self, epoch: int, price: float) -> None:
        bucket = epoch - (epoch % self.minute_bar_seconds)
        if self._minute_bucket_start is None:
            self._minute_bucket_start = bucket
            self._minute_bucket_last_price = price
            return
        if bucket == self._minute_bucket_start:
            # still inside the current bucket -- track its running last
            # price, which becomes this bucket's close once it rolls over
            self._minute_bucket_last_price = price
            return
        # a new minute bucket has started: the PREVIOUS bucket's last price
        # is that minute bar's close
        close = self._minute_bucket_last_price
        if self._last_minute_close is not None and self._last_minute_close > 0 and close and close > 0:
            self.minute_log_returns.append(math.log(close / self._last_minute_close))
        self._last_minute_close = close
        self._minute_bucket_start = bucket
        self._minute_bucket_last_price = price
