"""
Stake progression, kept fully separate from the risk engine's hard limits so
a progression bug can never itself become a risk-limit bypass.

IMPORTANT CONTEXT FROM PRIOR BOTS: martingale progression was added to a
sibling bot (digit_over_bot) and, over a 278-trade live sample, the top
martingale stake tier lost the most money with no better win rate than the
flat-stake tier -- it amplified losses rather than compensating for them,
because the underlying per-trade edge didn't hold up. Astra defaults
staking.enabled=false (flat stake) for exactly that reason. Progression is
still implemented and available behind a config flag for anyone who wants to
re-test it, but it is not the default and should not be enabled without
looking at Astra's own live edge data first.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StakingState:
    consecutive_losses: int = 0
    step: int = 0


class StakingEngine:
    def __init__(self, base_stake: float, enabled: bool, progression_factor: float,
                 max_steps: int, max_stake: float):
        self.base_stake = base_stake
        self.enabled = enabled
        self.progression_factor = progression_factor
        self.max_steps = max_steps
        self.max_stake = max_stake
        self._state: dict[str, StakingState] = {}

    def _get(self, symbol: str) -> StakingState:
        return self._state.setdefault(symbol, StakingState())

    def current_stake(self, symbol: str) -> float:
        if not self.enabled:
            return self.base_stake
        state = self._get(symbol)
        stake = self.base_stake * (self.progression_factor ** state.step)
        return min(stake, self.max_stake)

    def record_result(self, symbol: str, won: bool) -> None:
        state = self._get(symbol)
        if won:
            state.step = 0
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1
            if self.enabled and state.step < self.max_steps:
                state.step += 1
            elif self.enabled:
                state.step = 0  # reset after max steps -- never climb forever
