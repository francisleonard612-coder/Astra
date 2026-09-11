from __future__ import annotations

from dataclasses import dataclass

from pricing.breakeven import breakeven_probability, loss_if_lose, profit_if_win
from pricing.payout import ContractQuote


@dataclass
class EdgeResult:
    quote: ContractQuote
    calibrated_probability: float
    breakeven_probability: float
    edge: float
    expected_value: float


def compute_edge(quote: ContractQuote, calibrated_probability: float) -> EdgeResult:
    be = breakeven_probability(quote)
    edge = calibrated_probability - be
    ev = calibrated_probability * profit_if_win(quote) - (1 - calibrated_probability) * loss_if_lose(quote)
    return EdgeResult(
        quote=quote,
        calibrated_probability=calibrated_probability,
        breakeven_probability=be,
        edge=edge,
        expected_value=ev,
    )
