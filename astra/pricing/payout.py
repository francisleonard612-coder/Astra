"""
Fetches live contract economics (stake, payout) from Deriv for a given
symbol/contract/barrier. Never assumes a fixed payout -- every trade
evaluation calls this fresh, immediately before the buy decision, and the
execution engine re-validates the proposal hasn't gone stale before buying
(see execution/orders.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from ingestion.deriv_client import DerivClient, DerivRequestError
from app.logging_setup import get_logger

logger = get_logger("pricing.payout")


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
                     stake: float, duration: int, duration_unit: str, currency: str) -> ContractQuote | None:
    try:
        proposal = await client.get_proposal(
            symbol=symbol, contract_type=contract_type, barrier=barrier,
            stake=stake, duration=duration, duration_unit=duration_unit, currency=currency,
        )
    except DerivRequestError as exc:
        logger.warning("Proposal request failed", extra={"extra_fields": {
            "symbol": symbol, "contract_type": contract_type, "barrier": barrier, "error": str(exc),
        }})
        return None

    if not proposal or "payout" not in proposal:
        return None

    return ContractQuote(
        symbol=symbol,
        contract_type=contract_type,
        barrier=barrier,
        stake=stake,
        payout=float(proposal["payout"]),
        ask_price=float(proposal.get("ask_price", stake)),
        proposal_id=proposal.get("id"),
        spot=float(proposal["spot"]) if proposal.get("spot") is not None else None,
    )
