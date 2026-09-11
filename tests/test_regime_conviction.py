from decision.regime_conviction import (
    REGIME_NEUTRAL, REGIME_RANGE_QUIET, REGIME_RANGE_VOLATILE,
    REGIME_TREND_QUIET, REGIME_TREND_VOLATILE, TRADEABLE_REGIMES,
    classify_regime, compute_conviction, conviction_outcome_report,
    conviction_stake, regime_decision, regime_votes,
)


def test_classify_regime_trend_quiet():
    regime, _ = classify_regime(hurst=0.65, sigma_now=1.0, sigma_baseline=1.0)
    assert regime == REGIME_TREND_QUIET


def test_classify_regime_trend_volatile():
    regime, _ = classify_regime(hurst=0.65, sigma_now=2.0, sigma_baseline=1.0)
    assert regime == REGIME_TREND_VOLATILE


def test_classify_regime_range_quiet():
    regime, _ = classify_regime(hurst=0.30, sigma_now=1.0, sigma_baseline=1.0)
    assert regime == REGIME_RANGE_QUIET


def test_classify_regime_range_volatile_is_skipped():
    regime, _ = classify_regime(hurst=0.30, sigma_now=2.0, sigma_baseline=1.0)
    assert regime == REGIME_RANGE_VOLATILE
    assert regime not in TRADEABLE_REGIMES


def test_classify_regime_neutral_band_is_skipped():
    regime, _ = classify_regime(hurst=0.50, sigma_now=1.0, sigma_baseline=1.0)
    assert regime == REGIME_NEUTRAL
    assert regime not in TRADEABLE_REGIMES


def test_classify_regime_no_baseline_yet():
    regime, why = classify_regime(hurst=0.65, sigma_now=1.0, sigma_baseline=0.0)
    assert regime == REGIME_NEUTRAL
    assert "baseline" in why


def test_regime_votes_filters_to_only_the_regime_layer_set():
    votes = {"a": 0.5, "b": -0.3, "c": 0.1, "d": 0.9}
    layers = {"TREND_QUIET": ["a", "c"]}
    filtered = regime_votes(votes, "TREND_QUIET", layers)
    assert filtered == {"a": 0.5, "c": 0.1}


def test_regime_votes_empty_for_unmapped_regime():
    votes = {"a": 0.5}
    assert regime_votes(votes, "SOME_OTHER_REGIME", {}) == {}


def test_compute_conviction_empty_votes():
    conviction, direction, _ = compute_conviction({})
    assert conviction == 0.0 and direction == 0


def test_compute_conviction_all_abstain():
    conviction, direction, reason = compute_conviction({"a": 0.0, "b": 0.0, "c": 0.0})
    assert conviction == 0.0 and direction == 0
    assert "neutral" in reason  # mean of all-zero votes is exactly zero -- hits that branch first


def test_compute_conviction_requires_minimum_voters():
    conviction, direction, reason = compute_conviction(
        {"a": 0.9, "b": 0.9}, cfg={"conviction_min_voters": 3})
    assert conviction == 0.0 and direction == 0
    assert "not a consensus" in reason


def test_compute_conviction_strong_unanimous_agreement():
    votes = {"a": 0.9, "b": 0.9, "c": 0.9}
    conviction, direction, _ = compute_conviction(votes)
    assert direction == 1
    assert conviction > 0.9  # unanimous, all near-max magnitude -> near-perfect conviction


def test_compute_conviction_split_disagreement_scores_low():
    votes = {"a": 0.9, "b": -0.9, "c": 0.1}
    conviction, direction, _ = compute_conviction(votes)
    assert conviction < 0.5  # violently split -- low agreement should crush conviction


def test_strength_normalization_bugfix_reproduces_source_documented_scenario():
    """Direct reproduction of the exact bug documented in
    compute_conviction()'s docstring: with the SOURCE's original raw
    |mean(votes)| strength (no normalization by the loudest voter), votes
    in the realistic +-0.06 to +-0.54 range the source measured live made a
    0.35 conviction floor mathematically unreachable -- 0 trades out of 276
    opportunities. The FIXED version (normalizing by max magnitude) must
    make a genuinely strong, unanimous small-magnitude vote set clear a
    reasonable floor."""
    # unanimous, realistic small-magnitude votes (well within the source's
    # measured +-0.06 to +-0.54 live range)
    votes = {"a": 0.30, "b": 0.28, "c": 0.25, "d": 0.31}

    # reproduce the OLD (buggy) unnormalized strength calculation directly,
    # to prove it really would have crushed conviction to near-zero
    vals = list(votes.values())
    mean_vote = sum(vals) / len(vals)
    old_buggy_strength = abs(mean_vote)  # no normalization by max magnitude
    assert old_buggy_strength < 0.35  # confirms: unreachable under the old floor

    conviction, direction, _ = compute_conviction(votes)
    assert direction == 1
    assert conviction > 0.9  # fixed version: near-perfect conviction on unanimous agreement
    assert conviction >= 0.35  # clears the floor the old version made unreachable


def test_conviction_stake_below_floor_returns_zero():
    stake, reason = conviction_stake(0.10, base_stake=1.0, cfg={"conviction_floor": 0.20})
    assert stake == 0.0
    assert "no trade" in reason


def test_conviction_stake_scales_linearly_between_floor_and_max():
    cfg = {"conviction_floor": 0.20, "conviction_min_mult": 0.5, "conviction_max_mult": 3.0}
    stake_at_floor, _ = conviction_stake(0.20, base_stake=1.0, cfg=cfg)
    stake_at_max, _ = conviction_stake(1.0, base_stake=1.0, cfg=cfg)
    stake_mid, _ = conviction_stake(0.60, base_stake=1.0, cfg=cfg)
    assert stake_at_floor == 0.5
    assert stake_at_max == 3.0
    assert stake_at_floor < stake_mid < stake_at_max


def test_conviction_stake_respects_max_stake_cap():
    stake, _ = conviction_stake(1.0, base_stake=10.0, cfg={"conviction_max_mult": 3.0, "conviction_max_stake": 5.0})
    assert stake == 5.0


def test_regime_decision_non_tradeable_regime_returns_no_trade_with_reason():
    result = regime_decision(
        hurst=0.50, sigma_now=1.0, sigma_baseline=1.0,  # neutral band
        layer_votes={"a": 0.9}, all_layer_names=["a"], regime_layers={},
        base_stake=1.0,
    )
    assert result["trade"] is False
    assert result["regime"] == REGIME_NEUTRAL
    assert any("not tradeable" in r for r in result["reasons"])


def test_regime_decision_full_path_produces_a_stake():
    layer_votes = {"a": 0.5, "b": 0.5, "c": 0.5, "d": -0.1}  # d is a different regime's layer
    regime_layers = {REGIME_TREND_QUIET: ["a", "b", "c"]}
    result = regime_decision(
        hurst=0.65, sigma_now=1.0, sigma_baseline=1.0,  # -> TREND_QUIET
        layer_votes=layer_votes, all_layer_names=["a", "b", "c", "d"],
        regime_layers=regime_layers, base_stake=1.0,
    )
    assert result["regime"] == REGIME_TREND_QUIET
    assert result["trade"] is True
    assert result["direction"] == 1
    assert result["stake"] > 0
    # "d" must be excluded -- it belongs to a different regime's layer set
    assert "d" not in [r for r in result["reasons"] if "active layers" in r][0]


def test_conviction_outcome_report_bins_by_conviction():
    trades = [{"conviction": c, "won": c > 0.5} for c in [0.1, 0.2, 0.6, 0.7, 0.9, 0.95]]
    report = conviction_outcome_report(trades, buckets=3)
    assert "win-rate check" in report
    assert "6 trades" in report


def test_conviction_outcome_report_empty():
    assert conviction_outcome_report([]) == "no completed trades yet"
