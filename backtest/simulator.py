"""
Tick-by-tick replay simulator.

SCOPE NOTE: this replays only the "global" architecture (SymbolPipeline),
not the full 3-way architecture competition added in
learning/architecture_competition.py -- it's still useful for sanity-
checking the core feature/model/decision pipeline offline (does it abstain
on noise, does it trade and win on an injected bias), just not for
comparing the three architectures against each other. That comparison
already happens live, continuously, in production via shadow evaluation;
extending this offline replay to cover it too is a reasonable follow-up but
wasn't in scope for this pass.

Reuses the exact same SymbolPipeline / DecisionEngine code paths as live
trading -- the only difference is that contract quotes are synthesized from
a configurable assumed payout instead of fetched from Deriv (there's no live
proposal to query in a replay), and no real orders are placed. This
guarantees no look-ahead: predictions at tick t are built only from
`state.window(...)` as of tick t, and the outcome used for `observe()` is
always the very next tick in the replayed sequence.

Usage:
    python -m backtest.simulator --digits path/to/digits.json --symbol R_100
or import `replay()` directly and pass a list[int] of digits (e.g. pulled
from astra_ticks via database/repository.py for a symbol/time range).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from app.config import get_config
from decision.decision_engine import SymbolPipeline
from models.ensemble import combine
from pricing.breakeven import breakeven_probability
from pricing.edge import compute_edge
from pricing.payout import ContractQuote
from pricing.mispricing import check_mispricing
from state.rolling_state import StateManager


@dataclass
class SimResult:
    n_ticks: int
    n_trades: int
    wins: int
    losses: int
    total_pnl: float
    final_champion_log_loss: float | None
    final_challenger_log_loss: float | None


def _synthetic_quote(symbol: str, contract_type: str, barrier: int, stake: float, assumed_payout_ratio: float) -> ContractQuote:
    """Approximates a Deriv payout for offline replay, WITHOUT a live proposal.

    IMPORTANT LIMITATION: real Deriv Over/Under payouts are priced per
    barrier -- a barrier of 2 (digit > 2, ~70% win probability on a fair
    digit) is priced with a correspondingly low payout, while a barrier of 8
    (~10% win probability) is priced with a much higher one, so the
    exchange's implied breakeven tracks the barrier's own base-rate geometry
    everywhere. This function does NOT model that: it applies one flat
    `assumed_payout_ratio` regardless of barrier, which means a backtest can
    show a "profitable edge" that is actually just the barrier's built-in
    win probability against an unrealistically generous flat payout -- not a
    real statistical edge over the market. Treat backtest P&L as a sanity
    check that the wiring (features -> models -> decision -> settlement)
    behaves as intended, never as an estimate of real returns. Live trading
    never uses this function -- pricing/payout.py always fetches a real
    proposal from Deriv before every decision and again before every buy.
    """
    payout = stake * (1 + assumed_payout_ratio)
    return ContractQuote(symbol=symbol, contract_type=contract_type, barrier=barrier, stake=stake,
                          payout=payout, ask_price=stake, proposal_id=None, spot=None)


def replay(symbol: str, digits: list[int], cfg=None, assumed_payout_ratio: float = 0.9,
           stake: float = 1.0) -> SimResult:
    cfg = cfg or get_config()
    over_barrier = cfg.get("contracts", "over_barrier", default=2)
    under_barrier = cfg.get("contracts", "under_barrier", default=7)
    min_samples = cfg.get("min_samples_per_symbol", default=300)
    mp_cfg = cfg.get("mispricing", default={})

    max_window = max(cfg.get("feature_windows", default=[2500]))
    max_markov_order = cfg.get("max_markov_order", default=3)
    state_manager = StateManager(max_window=max_window, max_markov_order=max_markov_order)
    state = state_manager.get(symbol)
    pipeline = SymbolPipeline(symbol, cfg)

    n_trades = wins = losses = 0
    total_pnl = 0.0
    pending = None

    for i, digit in enumerate(digits):
        if pending is not None:
            predictions, bundle, _over_p, _under_p = pending
            pipeline.observe(state, bundle, predictions, digit, over_barrier, under_barrier)

        state.push(digit)

        if not state.has_min_samples(min_samples):
            predictions, bundle = pipeline.predict(state)
            pending = (predictions, bundle, 0.0, 0.0)
            continue

        predictions, bundle = pipeline.predict(state)
        ensemble_vec = combine(predictions, pipeline.champion_weights)
        raw_over = float(np.sum(ensemble_vec[over_barrier + 1:]))
        raw_under = float(np.sum(ensemble_vec[:under_barrier]))
        cal_over = pipeline.calibration_over.calibrate(raw_over)
        cal_under = pipeline.calibration_under.calibrate(raw_under)
        pipeline._pending_over_prob = raw_over
        pipeline._pending_under_prob = raw_under

        quote_over = _synthetic_quote(symbol, "DIGITOVER", over_barrier, stake, assumed_payout_ratio)
        quote_under = _synthetic_quote(symbol, "DIGITUNDER", under_barrier, stake, assumed_payout_ratio)
        edge_over = compute_edge(quote_over, cal_over)
        edge_under = compute_edge(quote_under, cal_under)

        best = None
        for side, edge_result, cal_score in (
            ("OVER", edge_over, pipeline.calibration_over.quality_score()),
            ("UNDER", edge_under, pipeline.calibration_under.quality_score()),
        ):
            check = check_mispricing(
                edge_result, sample_size=state.total_observed, calibration_score=cal_score,
                model_agreement=0.8,  # not recomputed in this lightweight replay
                minimum_edge=mp_cfg.get("minimum_edge", 0.03),
                minimum_probability=mp_cfg.get("minimum_probability", 0.55),
                minimum_calibration_score=mp_cfg.get("minimum_calibration_score", 0.6),
                minimum_model_agreement=0.0,
                minimum_sample_size=mp_cfg.get("minimum_sample_size", 300),
            )
            if check.passes and (best is None or edge_result.expected_value > best[1].expected_value):
                best = (side, edge_result)

        if i + 1 < len(digits) and best is not None:
            side, edge_result = best
            next_digit = digits[i + 1]
            won = (next_digit > over_barrier) if side == "OVER" else (next_digit < under_barrier)
            n_trades += 1
            pnl = edge_result.quote.payout - stake if won else -stake
            total_pnl += pnl
            wins += int(won)
            losses += int(not won)

        pending = (predictions, bundle, raw_over, raw_under)

    return SimResult(
        n_ticks=len(digits), n_trades=n_trades, wins=wins, losses=losses, total_pnl=total_pnl,
        final_champion_log_loss=(
            float(np.mean(pipeline._ensemble_logloss_champion)) if pipeline._ensemble_logloss_champion else None
        ),
        final_challenger_log_loss=(
            float(np.mean(pipeline._ensemble_logloss_challenger)) if pipeline._ensemble_logloss_challenger else None
        ),
    )


def _main():
    parser = argparse.ArgumentParser(description="Replay a digit sequence through Astra's pipeline")
    parser.add_argument("--digits", required=True, help="Path to a JSON file containing a list of ints")
    parser.add_argument("--symbol", default="R_100")
    parser.add_argument("--stake", type=float, default=1.0)
    args = parser.parse_args()

    print("NOTE: this replay uses a flat placeholder payout, not real per-barrier "
          "Deriv pricing -- see the _synthetic_quote docstring. Treat P&L here as a "
          "pipeline sanity check, not a real performance estimate.\n")

    with open(args.digits) as f:
        digits = json.load(f)

    result = replay(args.symbol, digits, stake=args.stake)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    _main()
