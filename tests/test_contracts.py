from pricing.contracts import RISE, FALL, verify_contract_direction


def test_agreeing_longcode_passes():
    assert verify_contract_direction({"longcode": "USD 10.00 payout if Volatility 100 Index rises"}, RISE)
    assert verify_contract_direction({"longcode": "USD 10.00 payout if Volatility 100 Index falls"}, FALL)


def test_disagreeing_longcode_fails():
    assert not verify_contract_direction({"longcode": "USD 10.00 payout if Volatility 100 Index falls"}, RISE)
    assert not verify_contract_direction({"longcode": "USD 10.00 payout if Volatility 100 Index rises"}, FALL)


def test_missing_or_uninformative_longcode_does_not_block():
    assert verify_contract_direction({}, RISE)
    assert verify_contract_direction({"longcode": ""}, FALL)
    assert verify_contract_direction({"longcode": "USD 10.00 payout if digit matches"}, RISE)


def test_non_rise_fall_contract_types_always_pass():
    assert verify_contract_direction({"longcode": "USD 10.00 payout if Volatility 100 Index falls"}, "DIGITOVER")
