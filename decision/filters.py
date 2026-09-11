from __future__ import annotations

UNSTABLE_REGIMES = {"UNSTABLE", "UNKNOWN", "MODEL_DISAGREEMENT"}


def regime_blocks_trading(regime: str, allow_low_confidence_regimes: bool) -> bool:
    if allow_low_confidence_regimes:
        return False
    return regime in UNSTABLE_REGIMES


def collect_abstention_reasons(*, regime: str, allow_low_confidence_regimes: bool,
                                risk_ok: bool, risk_reason: str | None,
                                quote_available: bool, quality_score: float,
                                minimum_quality_score: float) -> list[str]:
    reasons: list[str] = []
    if not quote_available:
        reasons.append("stale_or_unavailable_contract_quote")
    if regime_blocks_trading(regime, allow_low_confidence_regimes):
        reasons.append(f"unstable_regime:{regime}")
    if not risk_ok:
        reasons.append(f"risk_limit:{risk_reason}")
    if quality_score < minimum_quality_score:
        reasons.append("quality_score_below_minimum")
    return reasons
