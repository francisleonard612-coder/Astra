"""
Rise/Fall contract-type mapping.

Deriv's OWN documentation (developers.deriv.com/docs/risefall) states this
backwards -- "For Rise, use PUT" / "For Fall, use CALL". Confirmed wrong:
two independently-built, live-trading bots on this account
(Rise-and-Fall-Minutes and Rise-and-Fall-Pro) both map UP/Rise -> CALL and
DOWN/Fall -> PUT -- the standard convention used everywhere else in options
trading, including Deriv's own docs for other contract types -- and that's
empirically the mapping that places the intended trade. Treat the docs page
as wrong, not this module.
"""
from __future__ import annotations

RISE = "CALL"
FALL = "PUT"

# Deriv's "allow the exit spot to equal the entry spot and still count as a
# win" variants. Off by default -- this changes the win condition, not
# something to reach for without deciding to on purpose.
RISE_EQUALS = "CALLE"
FALL_EQUALS = "PUTE"

RISE_FALL_CONTRACT_TYPES = {RISE, FALL, RISE_EQUALS, FALL_EQUALS}

_RISE_TYPES = {RISE, RISE_EQUALS}
_FALL_TYPES = {FALL, FALL_EQUALS}


def verify_contract_direction(proposal: dict, expected_contract_type: str) -> bool:
    """Best-effort runtime tripwire, not a hard gate.

    Deriv's proposal response carries a human-readable `longcode` describing
    the contract in plain English (e.g. "USD 10.00 payout if Volatility 100
    Index rises ..."). Cross-checking that text against the contract_type we
    actually asked for means a future change to Deriv's API (or a mistake in
    this module) fails LOUD on the very next quote instead of silently
    trading backwards forever -- exactly the failure mode a doc page being
    wrong about this exact mapping makes plausible.

    Returns True when it can't tell either way (no longcode, or a longcode
    that doesn't mention either direction) -- a missing signal should never
    block trading, only one that actively DISAGREES should.
    """
    if expected_contract_type not in RISE_FALL_CONTRACT_TYPES:
        return True

    longcode = (proposal.get("longcode") or "").lower()
    if not longcode:
        return True

    says_rise = "rise" in longcode or "higher" in longcode
    says_fall = "fall" in longcode or "lower" in longcode

    if expected_contract_type in _RISE_TYPES and says_fall and not says_rise:
        return False
    if expected_contract_type in _FALL_TYPES and says_rise and not says_fall:
        return False
    return True
