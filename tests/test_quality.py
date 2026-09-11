from decision.quality import score_trade
from pricing.edge import compute_edge
from pricing.payout import ContractQuote


def make_edge_result(calibrated_probability=0.7):
    quote = ContractQuote(symbol="R_100", contract_type="DIGITOVER", barrier=2, stake=1.0,
                           payout=1.9, ask_price=1.0, proposal_id="p1", spot=None)
    return compute_edge(quote, calibrated_probability)


def test_quality_score_higher_for_better_inputs():
    good = score_trade(make_edge_result(0.8), calibration_score=0.9, model_agreement=0.9,
                        regime="NORMAL", sample_size=1000, target_sample_size=900)
    bad = score_trade(make_edge_result(0.56), calibration_score=0.4, model_agreement=0.4,
                       regime="UNSTABLE", sample_size=50, target_sample_size=900)
    assert good.score > bad.score


def test_quality_score_bounded_0_to_100():
    result = score_trade(make_edge_result(0.99), calibration_score=1.0, model_agreement=1.0,
                          regime="NORMAL", sample_size=100000, target_sample_size=900)
    assert 0.0 <= result.score <= 100.0
