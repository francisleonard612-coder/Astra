import pytest

from pricing.breakeven import breakeven_probability, loss_if_lose, profit_if_win
from pricing.edge import compute_edge
from pricing.mispricing import check_mispricing
from pricing.payout import ContractQuote


def make_quote(stake=1.0, payout=1.9):
    return ContractQuote(symbol="R_100", contract_type="DIGITOVER", barrier=2, stake=stake,
                          payout=payout, ask_price=stake, proposal_id="p1", spot=None)


def test_breakeven_probability_matches_stake_over_payout():
    quote = make_quote(stake=1.0, payout=2.0)
    assert breakeven_probability(quote) == 0.5


def test_profit_and_loss_amounts():
    quote = make_quote(stake=1.0, payout=1.9)
    assert profit_if_win(quote) == pytest.approx(0.9)
    assert loss_if_lose(quote) == 1.0


def test_positive_edge_when_calibrated_prob_above_breakeven():
    quote = make_quote(stake=1.0, payout=1.9)  # breakeven ~ 0.526
    result = compute_edge(quote, calibrated_probability=0.65)
    assert result.edge > 0
    assert result.expected_value > 0


def test_negative_edge_when_calibrated_prob_below_breakeven():
    quote = make_quote(stake=1.0, payout=1.9)
    result = compute_edge(quote, calibrated_probability=0.4)
    assert result.edge < 0
    assert result.expected_value < 0


def test_mispricing_check_fails_on_low_sample_size_even_with_good_edge():
    quote = make_quote(stake=1.0, payout=1.9)
    edge_result = compute_edge(quote, calibrated_probability=0.8)
    check = check_mispricing(
        edge_result, sample_size=10, calibration_score=0.9, model_agreement=0.9,
        minimum_edge=0.03, minimum_probability=0.55, minimum_calibration_score=0.6,
        minimum_model_agreement=0.6, minimum_sample_size=300,
    )
    assert not check.passes
    assert "insufficient_sample_size" in check.reasons_failed


def test_mispricing_check_passes_when_all_gates_clear():
    quote = make_quote(stake=1.0, payout=1.9)
    edge_result = compute_edge(quote, calibrated_probability=0.8)
    check = check_mispricing(
        edge_result, sample_size=1000, calibration_score=0.9, model_agreement=0.9,
        minimum_edge=0.03, minimum_probability=0.55, minimum_calibration_score=0.6,
        minimum_model_agreement=0.6, minimum_sample_size=300,
    )
    assert check.passes
    assert check.reasons_failed == []
