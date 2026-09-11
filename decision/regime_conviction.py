"""
REGIME-CONDITIONAL ROUTING + ASYMMETRIC CONVICTION SIZING

Ported from regime_conviction.py. The philosophy (below, condensed from the
source's own extensive module docstring, which is worth reading in full if
you have the original file): classify the market regime FIRST, then consult
ONLY the signals whose assumptions hold in that regime -- not down-weighted,
IGNORED entirely, since a signal that's structurally wrong for current
conditions contributes biased noise, not just extra noise. Then size
position by CONVICTION (how strongly and how unanimously the regime-
appropriate signals agree), not flat-staking every qualifying trade --
concentrate capital in rare high-conviction setups, skip (not
downsize -- SKIP) marginal ones entirely. This is explicitly NOT martingale:
conviction is a property of the PRESENT read, indifferent to what the
previous trade did.

classify_regime(), compute_conviction(), conviction_stake(), and
conviction_outcome_report() port FAITHFULLY, unchanged -- they were already
fully generic in the source (a dict of named votes, not a hardcoded list),
including two real bugfixes already baked in there (see compute_conviction's
docstring for the strength-normalization one, found from 276 live trade
opportunities that produced exactly zero trades).

regime_votes() and regime_decision() are GENERALIZED versions of the
source's functions of the same name: the source hardcoded LAYER_NAMES (17
specific signals: markov, hmm, hawkes, ou, hurst, arfima, kalman, copula,
rsi, srsi, adx, boll, zscore, te, jump_dir, post_jump, sr) and REGIME_LAYERS
(which of those belong to which regime) as module constants. Most of those
are the technical-indicator grab-bag already flagged as not worth porting,
or signals Astra doesn't have fitted anywhere yet. Astra's actual Rise/Fall
signal set (mc_win_probability, calibration quality, drift-degraded state,
meta-learner output, ...) is a different shape entirely and doesn't map
onto the source's names. So here, `all_layer_names` and `regime_layers` are
parameters the caller supplies instead of module constants -- deciding
Astra's own layer-to-regime mapping (once real named Rise/Fall signals
exist to route) is deliberately left as a decision for whoever wires this
in, not something to default silently.

CAVEATS, unchanged from the source and just as true here:
  - The layer-to-regime map (whatever Astra's ends up being) is REASONED,
    not validated by walk-forward evidence. Treat it as a hypothesis until
    live results say otherwise.
  - Regime thresholds (Hurst 0.45/0.55, vol multiple) are starting points.
  - Conviction sizing concentrates risk by design. It's a bet that
    conviction correlates with win rate -- if it doesn't, this makes
    outcomes worse than flat staking, not better. Measure it with
    conviction_outcome_report() before raising conviction_max_mult.
"""
from __future__ import annotations

REGIME_TREND_QUIET = "TREND_QUIET"
REGIME_TREND_VOLATILE = "TREND_VOLATILE"
REGIME_RANGE_QUIET = "RANGE_QUIET"
REGIME_RANGE_VOLATILE = "RANGE_VOLATILE"  # skip -- chop, no edge
REGIME_NEUTRAL = "NEUTRAL"                # skip -- no regime, no justified layer set

TRADEABLE_REGIMES = {REGIME_TREND_QUIET, REGIME_TREND_VOLATILE, REGIME_RANGE_QUIET}


def classify_regime(hurst: float, sigma_now: float, sigma_baseline: float,
                     cfg: dict | None = None) -> tuple[str, str]:
    """Returns (regime, human_readable_reason).

    hurst          : Hurst exponent (trending > range < persistence).
    sigma_now      : current realized volatility.
    sigma_baseline : this symbol's own recent median/typical volatility.
                     Self-relative rather than an absolute threshold, so
                     this works unchanged across symbols with wildly
                     different native vol scales (1HZ10V vs R_75 differ by
                     orders of magnitude).
    """
    cfg = cfg or {}
    h_trend = cfg.get("regime_hurst_trend", 0.55)
    h_range = cfg.get("regime_hurst_range", 0.45)
    vol_mult = cfg.get("regime_vol_multiple", 1.35)

    if sigma_baseline <= 0:
        return REGIME_NEUTRAL, "no vol baseline yet"

    vol_ratio = sigma_now / sigma_baseline
    volatile = vol_ratio >= vol_mult

    if h_range <= hurst <= h_trend:
        return (REGIME_NEUTRAL,
                f"H={hurst:.3f} in neutral band [{h_range}, {h_trend}] "
                f"-- no regime, no justified layer set")

    if hurst > h_trend:
        if volatile:
            return REGIME_TREND_VOLATILE, f"H={hurst:.3f} trending, vol={vol_ratio:.2f}x baseline"
        return REGIME_TREND_QUIET, f"H={hurst:.3f} trending, vol={vol_ratio:.2f}x baseline (quiet)"

    if volatile:
        return (REGIME_RANGE_VOLATILE,
                f"H={hurst:.3f} anti-persistent with vol={vol_ratio:.2f}x baseline -- chop, no edge")
    return REGIME_RANGE_QUIET, f"H={hurst:.3f} anti-persistent, vol={vol_ratio:.2f}x baseline (quiet)"


def regime_votes(layer_votes: dict[str, float], regime: str,
                  regime_layers: dict[str, list[str]]) -> dict[str, float]:
    """Filters a full named-vote dict down to only the layers this regime's
    mapping says are relevant. Layers outside the set are dropped entirely
    -- not down-weighted. That's the whole point: a signal whose assumption
    doesn't currently hold contributes nothing rather than biased noise.

    Unlike the source's positional 17-float list keyed by a fixed
    LAYER_NAMES order, this takes a dict directly -- one fewer footgun (a
    reordered list silently mislabeling every vote) for a signal set that
    isn't fixed/known in advance the way the source's was.
    """
    wanted = set(regime_layers.get(regime, []))
    return {name: float(v) for name, v in layer_votes.items() if name in wanted}


def compute_conviction(votes: dict[str, float], cfg: dict | None = None) -> tuple[float, int, str]:
    """Returns (conviction, direction, reason). conviction in [0, 1],
    direction in {+1, -1, 0}.

    Conviction is the product of two things that must BOTH be high:
      strength  = normalised |mean(votes)| -- how strongly the layers lean
      agreement = fraction agreeing -- how unanimously they lean

    STRENGTH NORMALIZATION -- a real bug, fixed in the source after live
    data: raw |mean(votes)| assumes votes span roughly [-1, 1]. They don't.
    Measured across 276 live readings there, individual votes ran +-0.06 to
    +-0.54 and the resulting strength had a MEDIAN of 0.106 and MAXIMUM of
    0.334 -- multiplied by an agreement fraction (<=1.0), a conviction floor
    of 0.35 was mathematically unreachable: exactly zero trades out of 276
    opportunities, not rarely, never. Fix: divide the mean by the largest
    |vote| in the active set, so strength measures "how aligned relative to
    the loudest voter" and genuinely spans [0, 1].

    Multiplying strength by agreement (not averaging) is deliberate: a set
    that all agrees weakly (unanimous but near-zero) and a set that
    disagrees violently (strong but split) are BOTH low-conviction and
    should both size small. Averaging would let one mask the other.

    Votes of exactly 0.0 are abstentions -- excluded from the agreement
    denominator rather than counted as disagreement, since a neutral read
    isn't evidence against.
    """
    cfg = cfg or {}
    if not votes:
        return 0.0, 0, "no layers active for this regime"

    vals = list(votes.values())
    mean_vote = sum(vals) / len(vals)
    direction = 1 if mean_vote > 0 else (-1 if mean_vote < 0 else 0)
    if direction == 0:
        return 0.0, 0, "regime layers net exactly neutral"

    non_zero = [v for v in vals if abs(v) > 1e-9]
    if not non_zero:
        return 0.0, 0, "all regime layers abstained"

    # Require a minimum number of layers actually voting. Without this, a
    # single loud layer with the rest abstaining scores respectably (1 of 1
    # agreeing = 100% agreement, mean/max = 1.0 when it's the only voter) --
    # one layer is not a consensus, and sizing up on it defeats the point
    # of routing to a layer SET.
    min_voters = cfg.get("conviction_min_voters", 3)
    if len(non_zero) < min_voters:
        return 0.0, 0, (f"only {len(non_zero)} layer(s) voting, need >={min_voters} -- not a consensus")

    agreeing = sum(1 for v in non_zero if (v > 0) == (direction > 0))
    agreement = agreeing / len(non_zero)

    max_mag = max(abs(v) for v in non_zero)
    strength = min(1.0, abs(mean_vote) / max_mag) if max_mag > 0 else 0.0

    conviction = strength * agreement

    side = "CALL" if direction > 0 else "PUT"
    reason = (f"{side} strength={strength:.3f} (mean={abs(mean_vote):.3f} / "
              f"max_layer={max_mag:.3f}) agreement={agreement:.2f} "
              f"({agreeing}/{len(non_zero)} non-abstaining) conviction={conviction:.3f}")
    return conviction, direction, reason


def conviction_stake(conviction: float, base_stake: float, cfg: dict | None = None) -> tuple[float, str]:
    """Maps conviction -> stake. This is where the asymmetry happens.

    Below `floor`, returns 0.0 -- do not trade. Marginal reads are not
    small trades, they are no trades; taking them at any size is what
    erodes the account between the good setups.

    Above the floor, conviction is rescaled to [0,1] and mapped LINEARLY
    onto [min_mult, max_mult]. Linear rather than exponential is a
    deliberately conservative choice: exponential sizing on a signal whose
    conviction-to-win-rate correlation is UNPROVEN (see module docstring)
    would concentrate risk on an assumption not yet demonstrated.
    """
    cfg = cfg or {}
    floor = cfg.get("conviction_floor", 0.20)
    min_mult = cfg.get("conviction_min_mult", 0.5)
    max_mult = cfg.get("conviction_max_mult", 3.0)
    max_stake = cfg.get("conviction_max_stake", 0.0)  # 0 = uncapped

    if conviction < floor:
        return 0.0, (f"conviction {conviction:.3f} < floor {floor:.2f} "
                     f"-- no trade (marginal reads are skipped, not sized down)")

    span = max(1e-9, 1.0 - floor)
    scaled = (conviction - floor) / span
    mult = min_mult + scaled * (max_mult - min_mult)
    stake = base_stake * mult
    if max_stake > 0:
        stake = min(stake, max_stake)
    stake = round(stake, 2)
    return stake, f"conviction {conviction:.3f} -> {mult:.2f}x base -> stake ${stake:.2f}"


def regime_decision(hurst: float, sigma_now: float, sigma_baseline: float,
                     layer_votes: dict[str, float], all_layer_names: list[str],
                     regime_layers: dict[str, list[str]], base_stake: float,
                     cfg: dict | None = None) -> dict:
    """One call, the whole philosophy. Returns a dict with trade/regime/
    direction/conviction/stake/reasons. Every rejection path returns
    trade=False WITH the reason recorded, so logs always explain why
    nothing fired rather than going silent.

    `all_layer_names` and `regime_layers` are Astra's own signal set and
    layer-to-regime mapping (see module docstring) -- not defaulted here.
    """
    cfg = cfg or {}
    reasons = []

    regime, why = classify_regime(hurst, sigma_now, sigma_baseline, cfg)
    reasons.append(f"regime={regime} ({why})")

    if regime not in TRADEABLE_REGIMES:
        reasons.append("regime is not tradeable -- waiting")
        return {"trade": False, "regime": regime, "direction": 0,
                "conviction": 0.0, "stake": 0.0, "reasons": reasons}

    votes = regime_votes(layer_votes, regime, regime_layers)
    active = ", ".join(f"{k}={v:+.2f}" for k, v in votes.items())
    reasons.append(f"active layers ({len(votes)}): {active}")
    ignored = [n for n in all_layer_names if n not in votes]
    reasons.append(f"ignored ({len(ignored)}): {', '.join(ignored)}")

    conviction, direction, conv_why = compute_conviction(votes, cfg)
    reasons.append(conv_why)

    if direction == 0:
        return {"trade": False, "regime": regime, "direction": 0,
                "conviction": conviction, "stake": 0.0, "reasons": reasons}

    stake, stake_why = conviction_stake(conviction, base_stake, cfg)
    reasons.append(stake_why)

    return {"trade": stake > 0, "regime": regime, "direction": direction,
            "conviction": conviction, "stake": stake, "reasons": reasons}


def conviction_outcome_report(trades: list[dict], buckets: int = 4) -> str:
    """The whole approach rests on one unproven assumption: higher
    conviction means a higher win rate. If that correlation is absent,
    conviction sizing actively makes results WORSE than flat staking,
    because it puts more money on reads that are no better than average.

    Bins completed trades by conviction and reports win rate per bin, so
    the assumption gets checked against real outcomes instead of taken on
    faith. Feed it dicts with "conviction" and "won" keys.

    Read it as: win rate should climb roughly monotonically across buckets.
    Flat -> drop conviction_max_mult to 1.0 (flat staking) until it isn't.
    Inverted -> the layer-to-regime map is wrong for these symbols.
    """
    if not trades:
        return "no completed trades yet"

    scored = [t for t in trades if "conviction" in t and "won" in t]
    if not scored:
        return "no trades carry conviction+won fields"

    scored.sort(key=lambda t: t["conviction"])
    n = len(scored)
    size = max(1, n // buckets)
    lines = [f"Conviction -> win-rate check ({n} trades):"]
    for i in range(0, n, size):
        chunk = scored[i:i + size]
        if not chunk:
            continue
        wins = sum(1 for t in chunk if t["won"])
        lo = chunk[0]["conviction"]
        hi = chunk[-1]["conviction"]
        lines.append(f"  conviction {lo:.2f}-{hi:.2f}: {wins}/{len(chunk)} = "
                     f"{wins / len(chunk) * 100:.1f}% win rate")
    lines.append("  (want: win rate rising across buckets. Flat -> set "
                 "conviction_max_mult=1.0. Inverted -> layer map is wrong.)")
    return "\n".join(lines)
