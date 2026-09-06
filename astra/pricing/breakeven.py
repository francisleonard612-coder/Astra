from __future__ import annotations

from pricing.payout import ContractQuote


def breakeven_probability(quote: ContractQuote) -> float:
    """Probability of winning required to break even, derived from the live
    proposal, not an assumed fixed payout structure."""
    if quote.payout <= 0:
        return 1.0
    return quote.ask_price / quote.payout


def profit_if_win(quote: ContractQuote) -> float:
    return quote.payout - quote.ask_price


def loss_if_lose(quote: ContractQuote) -> float:
    return quote.ask_price
