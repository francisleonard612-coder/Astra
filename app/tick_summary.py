"""
Per-symbol rolling tick-outcome summary, logged (and persisted to
astra_system_events) every `window_size` ticks so a Railway log skim
answers "is Astra actually trading, and if not, why not" without having to
query Supabase or manually parse decision.reason strings.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_ARCH_PREFIX_RE = re.compile(r"^\[(global|specialist|hybrid)\]\s*")


def extract_no_trade_reasons(reason: str) -> list[str]:
    """A NO_TRADE decision.reason is either a single token
    ("insufficient_sample_size", "no_ready_models", "no_positive_edge") or a
    comma-joined set of gate failures, optionally prefixed with the
    architecture that produced it (e.g. "[global] insufficient_edge,
    probability_below_minimum"). This splits it into individual reason
    tokens so a summary counts how often each SPECIFIC gate is the blocker,
    rather than treating every distinct combination as its own bucket --
    "insufficient_edge appeared in 40/150 ticks" is a debuggable signal;
    "'insufficient_edge,quality_score_below_minimum' appeared in 12/150
    ticks, 'insufficient_edge' alone in 9/150, ..." is not, at a glance.
    """
    reason = _ARCH_PREFIX_RE.sub("", reason or "")
    return [tok.strip() for tok in reason.split(",") if tok.strip()]


@dataclass
class TickSummary:
    window_ticks: int
    trades_executed: int
    wins: int
    losses: int
    pnl: float
    no_trade_ticks: int
    top_no_trade_reasons: dict[str, int]
    champion_architecture: str
    sample_size: int


class TickSummaryTracker:
    """One instance per symbol. Feed it every tick's outcome via
    `record_trade` / `record_risk_blocked` / `record_no_trade`; check
    `due()` after each tick and call `build_and_reset()` when it fires."""

    def __init__(self, window_size: int = 150):
        self.window_size = window_size
        self._reset()

    def _reset(self) -> None:
        self.ticks = 0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.no_trade_reasons: Counter = Counter()

    def record_no_trade(self, reason: str) -> None:
        self.ticks += 1
        for token in extract_no_trade_reasons(reason):
            self.no_trade_reasons[token] += 1

    def record_risk_blocked(self, risk_reason: str | None) -> None:
        self.ticks += 1
        self.no_trade_reasons[f"risk_blocked:{risk_reason}"] += 1

    def record_trade(self, won: bool | None, pnl: float | None) -> None:
        self.ticks += 1
        self.trades += 1
        if won is True:
            self.wins += 1
        elif won is False:
            self.losses += 1
        if pnl is not None:
            self.pnl += pnl

    def due(self) -> bool:
        return self.ticks >= self.window_size

    def build_and_reset(self, champion_architecture: str, sample_size: int) -> TickSummary:
        summary = TickSummary(
            window_ticks=self.ticks,
            trades_executed=self.trades,
            wins=self.wins,
            losses=self.losses,
            pnl=round(self.pnl, 4),
            no_trade_ticks=self.ticks - self.trades,
            top_no_trade_reasons=dict(self.no_trade_reasons.most_common(5)),
            champion_architecture=champion_architecture,
            sample_size=sample_size,
        )
        self._reset()
        return summary
