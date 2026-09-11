from app.tick_summary import TickSummaryTracker, extract_no_trade_reasons


def test_extract_reasons_strips_architecture_prefix():
    assert extract_no_trade_reasons("[global] insufficient_edge") == ["insufficient_edge"]


def test_extract_reasons_splits_comma_joined_gates():
    reasons = extract_no_trade_reasons("[hybrid] insufficient_edge,probability_below_minimum")
    assert reasons == ["insufficient_edge", "probability_below_minimum"]


def test_extract_reasons_handles_bare_single_token():
    assert extract_no_trade_reasons("insufficient_sample_size") == ["insufficient_sample_size"]


def test_extract_reasons_handles_empty_or_none():
    assert extract_no_trade_reasons("") == []
    assert extract_no_trade_reasons(None) == []


def test_tracker_not_due_before_window_fills():
    tracker = TickSummaryTracker(window_size=150)
    for _ in range(149):
        tracker.record_no_trade("insufficient_sample_size")
    assert not tracker.due()
    tracker.record_no_trade("insufficient_sample_size")
    assert tracker.due()


def test_summary_counts_trades_and_reasons_correctly():
    tracker = TickSummaryTracker(window_size=10)
    for _ in range(6):
        tracker.record_no_trade("[global] insufficient_edge")
    for _ in range(2):
        tracker.record_no_trade("[global] quality_score_below_minimum")
    tracker.record_trade(won=True, pnl=0.9)
    tracker.record_trade(won=False, pnl=-1.0)

    assert tracker.due()
    summary = tracker.build_and_reset(champion_architecture="global", sample_size=1000)

    assert summary.window_ticks == 10
    assert summary.trades_executed == 2
    assert summary.wins == 1
    assert summary.losses == 1
    assert summary.pnl == -0.1
    assert summary.no_trade_ticks == 8
    assert summary.top_no_trade_reasons["insufficient_edge"] == 6
    assert summary.top_no_trade_reasons["quality_score_below_minimum"] == 2
    assert summary.champion_architecture == "global"
    assert summary.sample_size == 1000


def test_tracker_resets_after_building_summary():
    tracker = TickSummaryTracker(window_size=5)
    for _ in range(5):
        tracker.record_no_trade("insufficient_sample_size")
    tracker.build_and_reset("global", 100)
    assert tracker.ticks == 0
    assert not tracker.due()
    assert tracker.no_trade_reasons == {}


def test_risk_blocked_reason_is_labeled_distinctly():
    tracker = TickSummaryTracker(window_size=1)
    tracker.record_risk_blocked("max_concurrent_trades_reached")
    summary = tracker.build_and_reset("global", 500)
    assert summary.top_no_trade_reasons == {"risk_blocked:max_concurrent_trades_reached": 1}


def test_trade_with_unknown_outcome_still_counts_as_a_trade_attempt():
    tracker = TickSummaryTracker(window_size=1)
    tracker.record_trade(won=None, pnl=None)
    summary = tracker.build_and_reset("global", 500)
    assert summary.trades_executed == 1
    assert summary.wins == 0
    assert summary.losses == 0
    assert summary.pnl == 0.0
