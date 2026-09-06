"""
Risk engine.

The model layer decides whether an opportunity looks good. This module
decides whether the account is *allowed* to take it, and nothing the model
layer says can override that -- decision_engine.py only ever receives
`risk_ok=False` plus a reason; it can't ask the risk engine to make an
exception.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RiskState:
    session_start_equity: float | None = None
    peak_equity: float | None = None
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    day_key: str | None = None
    cooldown_until: float | None = None
    emergency_stop: bool = False
    connection_ok: bool = True


class RiskEngine:
    def __init__(self, base_stake: float, max_stake: float, max_consecutive_losses: int,
                 max_daily_loss: float, max_drawdown: float, max_trades_per_day: int,
                 cooldown_seconds_after_max_losses: int):
        self.base_stake = base_stake
        self.max_stake = max_stake
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_seconds = cooldown_seconds_after_max_losses
        self.state = RiskState()

    def _roll_day_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self.state.day_key != today:
            self.state.day_key = today
            self.state.realized_pnl_today = 0.0
            self.state.trades_today = 0

    def set_equity(self, equity: float) -> None:
        if self.state.session_start_equity is None:
            self.state.session_start_equity = equity
        if self.state.peak_equity is None or equity > self.state.peak_equity:
            self.state.peak_equity = equity
        self._current_equity = equity

    def set_connection_ok(self, ok: bool) -> None:
        self.state.connection_ok = ok

    def trigger_emergency_stop(self) -> None:
        self.state.emergency_stop = True

    def clear_emergency_stop(self) -> None:
        self.state.emergency_stop = False

    def check(self, stake: float) -> tuple[bool, str | None]:
        self._roll_day_if_needed()

        if self.state.emergency_stop:
            return False, "emergency_stop"
        if not self.state.connection_ok:
            return False, "connection_failure_stop"
        if self.state.cooldown_until and time.time() < self.state.cooldown_until:
            return False, "cooldown_active"
        if stake > self.max_stake:
            return False, "stake_exceeds_max_stake"
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.state.cooldown_until = time.time() + self.cooldown_seconds
            return False, "max_consecutive_losses_reached"
        if self.state.realized_pnl_today <= -abs(self.max_daily_loss):
            return False, "max_daily_loss_reached"
        if self.state.trades_today >= self.max_trades_per_day:
            return False, "max_trades_per_day_reached"
        if (self.state.peak_equity is not None and getattr(self, "_current_equity", None) is not None
                and self.state.peak_equity > 0):
            drawdown = self.state.peak_equity - self._current_equity
            if drawdown >= self.max_drawdown:
                return False, "max_drawdown_reached"

        return True, None

    def record_trade_result(self, pnl: float) -> None:
        self._roll_day_if_needed()
        self.state.trades_today += 1
        self.state.realized_pnl_today += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
