from pricing.duration_grid import DurationCandidate, build_candidate_grid, filter_to_allowed


def test_build_candidate_grid_spans_ticks_and_minutes():
    grid = build_candidate_grid(tick_durations=[5, 10], minute_durations=[1, 3])
    assert grid == [
        DurationCandidate(5, "t"), DurationCandidate(10, "t"),
        DurationCandidate(1, "m"), DurationCandidate(3, "m"),
    ]


def test_str_representation():
    assert str(DurationCandidate(5, "t")) == "5t"
    assert str(DurationCandidate(3, "m")) == "3m"


def test_filter_to_allowed_excludes_out_of_bounds_candidates():
    grid = build_candidate_grid(tick_durations=[1, 5, 10, 50], minute_durations=[1, 5, 60])
    contracts_for = {"available": [
        {"contract_type": "CALL", "min_contract_duration": "5t", "max_contract_duration": "10t"},
    ]}
    allowed = filter_to_allowed(grid, contracts_for, "CALL")
    # tick candidates within [5, 10] survive; out-of-range ticks are dropped;
    # minute candidates are untouched since the bound is tick-denominated
    assert DurationCandidate(1, "t") not in allowed
    assert DurationCandidate(5, "t") in allowed
    assert DurationCandidate(10, "t") in allowed
    assert DurationCandidate(50, "t") not in allowed
    assert DurationCandidate(1, "m") in allowed
    assert DurationCandidate(5, "m") in allowed
    assert DurationCandidate(60, "m") in allowed


def test_filter_to_allowed_passes_through_when_contract_type_absent():
    grid = build_candidate_grid(tick_durations=[5], minute_durations=[1])
    contracts_for = {"available": [{"contract_type": "DIGITOVER", "min_contract_duration": "1t"}]}
    assert filter_to_allowed(grid, contracts_for, "CALL") == grid


def test_filter_to_allowed_never_cross_compares_units():
    # A tick-denominated bound must never exclude a minute candidate, and
    # vice versa -- ticks and minutes aren't directly comparable without
    # knowing this symbol's exact tick interval.
    grid = [DurationCandidate(1, "m")]
    contracts_for = {"available": [
        {"contract_type": "CALL", "min_contract_duration": "5t", "max_contract_duration": "10t"},
    ]}
    assert filter_to_allowed(grid, contracts_for, "CALL") == grid
