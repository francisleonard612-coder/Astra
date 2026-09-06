"""
Fetches live contract economics (stake, payout) from Deriv for a given
symbol/contract/barrier. Never assumes a fixed payout -- every trade
evaluation calls this fresh, immediately before the buy decision, and the
execution engine re-validates the proposal hasn't gone stale before buying
(see execution/orders.py).

Short-TTL cache: Deriv's proposal/proposal_open_contract/buy/sell calls
share ONE 300-360/min budget per connection (developers.deriv.com/docs/limits).
Astra runs one worker per symbol, each evaluating on every tick and fetching
two quotes (OVER + UNDER) per evaluation -- with more than a handful of
symbols this blows through the budget in seconds even with the DerivClient
rate limiter throttling sends (observed in production: dozens of "You have
reached the rate limit for proposal" errors per second). Payout for a fixed
barrier/duration/stake doesn't meaningfully move tick-to-tick on a synthetic
index, so caching each (symbol, contract_type, barrier, stake, duration,
currency) combination for a few seconds cuts the vast majority of this
traffic with negligible staleness -- execute_decision() re-fetches a fresh
proposal right before buying regardless, so a stale cached quote here can
delay a trade by at most one cache TTL, never cause a trade at a stale price.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ingestion.deriv_client import DerivClient, DerivRequestError
from app.logging_setup import get_logger

logger = get_logger("pricing.payout")

QUOTE_CACHE_TTL_SECONDS = 5.0
_quote_cache: dict[tuple, tuple[float, "ContractQuote"]] = {}


@dataclass
class ContractQuote:
    symbol: str
    contract_type: str  # DIGITOVER | DIGITUNDER
    barrier: int
    stake: float
    payout: float
    ask_price: float
    proposal_id: str | None
    spot: float | None


async def get_quote(client: DerivClient, symbol: str, contract_type: str, barrier: int,
                     stake: float, duration: int, duration_unit: str, currency: str,
                     bypass_cache: bool = False) -> ContractQuote | None:
    cache_key = (symbol, contract_type, barrier, round(stake, 2), duration, duration_unit, currency)
    cached = _quote_cache.get(cache_key)
    now = time.monotonic()
    if not bypass_cache and cached is not None and (now - cached[0]) < QUOTE_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        proposal = await client.get_proposal(
            symbol=symbol, contract_type=contract_type, barrier=barrier,
            stake=stake, duration=duration, duration_unit=duration_unit, currency=currency,
        )
    except DerivRequestError as exc:
        logger.warning("Proposal request failed", extra={"extra_fields": {
            "symbol": symbol, "contract_type": contract_type, "barrier": barrier, "error": str(exc),
        }})
        # Serve a stale cached quote rather than nothing if we have one --
        # better to evaluate against a slightly-stale payout than to skip
        # the tick entirely because of a transient rate-limit rejection.
        # Never do this for the bypass_cache=True pre-buy re-validation call
        # in execution/orders.py -- that call exists specifically to refuse
        # a stale price, so falling back to a stale quote there would
        # silently defeat its own purpose.
        return cached[1] if (cached is not None and not bypass_cache) else None

    if not proposal or "payout" not in proposal:
        return None

    quote = ContractQuote(
        symbol=symbol,
        contract_type=contract_type,
        barrier=barrier,
        stake=stake,
        payout=float(proposal["payout"]),
        ask_price=float(proposal.get("ask_price", stake)),
        proposal_id=proposal.get("id"),
        spot=float(proposal["spot"]) if proposal.get("spot") is not None else None,
    )
    _quote_cache[cache_key] = (now, quote)
    return quote
