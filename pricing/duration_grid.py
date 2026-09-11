"""
Candidate trade durations spanning both ticks and minutes.

This module only builds and validates the CANDIDATE SET -- it does not rank
or select among them. Kept deliberately separate from any ranking/Monte
Carlo logic: risefall_bot_v4_hmm_gbm.py's monte_carlo_duration() found and
fixed three real bugs where projecting a noisy drift estimate forward by
`dur` (scaling O(dur)) while only scaling diffusion noise by sqrt(dur)
mechanically biased duration selection toward the LONGEST candidate on pure
noise. That fix belongs entirely inside the ranking function once it's
ported -- this module's only job is "what durations are even on the table",
so the two don't get tangled together.

Deriv's actual min/max duration per (symbol, contract_type) varies and can
change server-side -- see DerivClient.get_contracts_for. A static default
grid in config is only safe to trade because filter_to_allowed() below
cross-checks it against that live data before use, never because the
defaults themselves are assumed correct.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DurationCandidate:
    duration: int
    duration_unit: str  # "t" | "m"

    @property
    def approx_seconds(self) -> float:
        # Ticks on these synthetic indices average ~2s apart. Rough figure
        # for same-units comparisons only (e.g. sizing a settlement-wait
        # timeout) -- never used in the actual duration-selection math.
        return self.duration * (2.0 if self.duration_unit == "t" else 60.0)

    def __str__(self) -> str:
        return f"{self.duration}{self.duration_unit}"


def build_candidate_grid(tick_durations: list[int], minute_durations: list[int]) -> list[DurationCandidate]:
    grid = [DurationCandidate(d, "t") for d in tick_durations]
    grid += [DurationCandidate(d, "m") for d in minute_durations]
    return grid


def _parse_duration_string(s: str | None) -> DurationCandidate | None:
    # Deriv expresses contracts_for's min/max_contract_duration as strings
    # like "1t" / "1m" / "15s" / "1d" -- only "t" and "m" are ever produced
    # by build_candidate_grid, so anything else parses but will simply never
    # match a candidate's unit in _within_duration_bounds below.
    if not s:
        return None
    unit = s[-1]
    try:
        value = int(s[:-1])
    except ValueError:
        return None
    return DurationCandidate(value, unit)


def _within_duration_bounds(candidate: DurationCandidate, min_str: str | None, max_str: str | None) -> bool:
    lo, hi = _parse_duration_string(min_str), _parse_duration_string(max_str)
    # Only compare within the SAME unit. Deriv's bounds are expressed in
    # whatever unit that contract's limits happen to use, which isn't
    # necessarily convertible to a candidate's unit without knowing this
    # symbol's exact tick interval. Conservative default: a bound in a
    # different unit never excludes a candidate, rather than risk wrongly
    # rejecting a valid minute candidate against a tick-denominated bound
    # (or vice versa).
    if lo is not None and lo.duration_unit == candidate.duration_unit and candidate.duration < lo.duration:
        return False
    if hi is not None and hi.duration_unit == candidate.duration_unit and candidate.duration > hi.duration:
        return False
    return True


def filter_to_allowed(candidates: list[DurationCandidate], contracts_for: dict,
                       contract_type: str) -> list[DurationCandidate]:
    """Cross-check a candidate grid against a live `contracts_for` response
    for one symbol. Returns the input unchanged if that contract_type isn't
    present in the response at all -- callers decide separately whether
    trading without a confirmed limit is acceptable (it generally shouldn't
    be, for a contract type never seen from this call before)."""
    limits = None
    for row in contracts_for.get("available", []):
        if row.get("contract_type") == contract_type:
            limits = row
            break
    if limits is None:
        return candidates

    min_d = limits.get("min_contract_duration")
    max_d = limits.get("max_contract_duration")
    return [c for c in candidates if _within_duration_bounds(c, min_d, max_d)]
