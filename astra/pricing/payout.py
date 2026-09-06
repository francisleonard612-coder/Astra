"""
Fetches live contract economics (stake, payout) from Deriv for a given
symbol/contract/barrier. Never assumes a fixed payout -- every trade
evaluation calls this fresh, immediately before the buy decision, and the
execution engine re-validates the proposal hasn't gone stale before buying
(see execution/orders.py).

Short-TTL cache: Deriv's proposal/proposal_open_contract/buy/sell calls
share ONE 360/min + 14,400/hour budget per connection
(developers.deriv.com/docs/limits). Astra runs one worker per symbol, each
evaluating on every tick and fetching two quotes (OVER + UNDER) per
evaluation -- with more than a handful of symbols this blows through the
budget in seconds even with the DerivClient rate limiter throttling sends
(observed in production: dozens of "You have reached the rate limit for
proposal" errors per second). Payout for a fixed barrier/duration/stake
doesn't meaningfully move tick-to-tick on a synthetic index, so caching each
(symbol, contract_type, barrier, stake, duration, currency) combination for
a few seconds cuts the vast majority of this traffic with negligible
staleness -- execute_decision() re-fetches a fresh proposal right before
buying regardless, so a stale cached quote here can delay a trade by at most
one cache TTL, never cause a trade at a stale price.

Rate-limit backoff (the actual production bug this module previously had):
the cache above only ever gets POPULATED by a *successful* proposal call.
Once Deriv starts rejecting proposal requests with a RateLimit error --
whether because this bot's own traffic briefly exceeded the shared budget,
or because something else on the same account/connection (another bot,
proposal_open_contract polling from execution/orders.py, a burst of buys)
consumed it -- the old code logged the rejection and simply tried again on
the very next tick, roughly two seconds later. Deriv's own guidance
(developers.deriv.com/docs/best-practices) is explicit that rejected
requests must be backed off, not retried immediately, precisely because the
budget is a shared sliding window: a request that gets rejected still lands
in that window, so retrying every tick keeps re-arming the same limit and
the window never has a gap long enough to drain. That is exactly the
failure mode seen in production -- once the first proposal call of a run
was rejected, literally every subsequent call was rejected too, for the
entire lifetime of the log, because the bot never stopped adding fresh
requests to the same saturated window.

The fix is a short, escalating global cooldown (global, not per cache key,
because the rate limit is a single budget shared across every proposal
regardless of symbol/contract/barrier): a RateLimit rejection suspends new
proposal calls for a few seconds, doubling on repeated rejections up to a
cap, so the shared window actually gets a chance to empty out. A success
resets the cooldown immediately. While in cooldown, get_quote() serves the
last cached quote (never for the bypass_cache=True pre-buy revalidation
call, which must refuse to trade on a quote it can't confirm is current)
instead of adding another doomed request to the pile.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ingestion.deriv_client import DerivClient, DerivRequestError
from app.logging_setup import get_logger

logger = get_logger("pricing.payout")

QUOTE_CACHE_TTL_SECONDS = 5.0
_quote_cache: dict[tuple, tuple[float, "ContractQuote"]] = {}

# Backoff after a Deriv RateLimit rejection on the shared
# proposal/proposal_open_contract/buy/sell budget. Starts short (the budget
# is per-minute, so a few seconds is often enough once traffic stops adding
# to it) and doubles on consecutive rejections, capped well under a minute
# so a genuinely transient rejection doesn't wedge the bot for long.
RATE_LIMIT_BACKOFF_BASE_SECONDS = 3.0
RATE_LIMIT_BACKOFF_MAX_SECONDS = 45.0
_rate_limit_cooldown_until: float = 0.0
_consecutive_rate_limit_failures: int = 0


def _is_rate_limit_error(exc: DerivRequestError) -> bool:
    # Deriv's actual error object uses code == "RateLimit" (confirmed against
    # Deriv/Binary API error reports); fall back to a message match in case
    # the code is ever missing so an unrecognized-but-clearly-rate-limit
    # error still triggers backoff instead of being retried instantly.
    if getattr(exc, "code", None) == "RateLimit":
        return True
    return "rate limit" in str(exc).lower()


def _register_rate_limit_failure() -> None:
    global _rate_limit_cooldown_until, _consecutive_rate_limit_failures
    _consecutive_rate_limit_failures += 1
    backoff = min(
        RATE_LIMIT_BACKOFF_BASE_SECONDS * (2 ** (_consecutive_rate_limit_failures - 1)),
        RATE_LIMIT_BACKOFF_MAX_SECONDS,
    )
    _rate_limit_cooldown_until = time.monotonic() + backoff
    logger.warning("Backing off proposal requests after rate limit", extra={"extra_fields": {
        "backoff_seconds": backoff, "consecutive_failures": _consecutive_rate_limit_failures,
    }})


def _register_rate_limit_success() -> None:
    global _rate_limit_cooldown_until, _consecutive_rate_limit_failures
    _consecutive_rate_limit_failures = 0
    _rate_limit_cooldown_until = 0.0


def _in_rate_limit_cooldown(now: float) -> bool:
    return now < _rate_limit_cooldown_until


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

    # Still cooling down from a recent RateLimit rejection: don't add another
    # request to the same saturated budget. Serve the cache if we're allowed
    # to (never for bypass_cache -- that call exists specifically to refuse a
    # quote it can't confirm is fresh).
    if _in_rate_limit_cooldown(now):
        return cached[1] if (cached is not None and not bypass_cache) else None

    try:
        proposal = await client.get_proposal(
            symbol=symbol, contract_type=contract_type, barrier=barrier,
            stake=stake, duration=duration, duration_unit=duration_unit, currency=currency,
        )
    except DerivRequestError as exc:
        logger.warning("Proposal request failed", extra={"extra_fields": {
            "symbol": symbol, "contract_type": contract_type, "barrier": barrier, "error": str(exc),
        }})
        if _is_rate_limit_error(exc):
            _register_rate_limit_failure()
        # Serve a stale cached quote rather than nothing if we have one --
        # better to evaluate against a slightly-stale payout than to skip
        # the tick entirely because of a transient rate-limit rejection.
        # Never do this for the bypass_cache=True pre-buy re-validation call
        # in execution/orders.py -- that call exists specifically to refuse
        # a stale price, so falling back to a stale quote there would
        # silently defeat its own purpose.
        return cached[1] if (cached is not None and not bypass_cache) else None

    _register_rate_limit_success()

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
