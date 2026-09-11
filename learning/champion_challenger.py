"""
Champion/challenger, scoped pragmatically for this build.

The spec's full vision (section 12) is separate champion/challenger *model
architectures* per digit specialist. What's implemented here instead is
champion/challenger over the ENSEMBLE WEIGHT VECTOR per symbol -- the
production ensemble always predicts with `pipeline.champion_weights`
(promoted, stable) while `pipeline.performance.current_weights()` (the
challenger, driven by rolling log-loss) is continuously compared against it
on the same realized outcomes. This keeps the "champion is what's actually
used in production, a challenger has to earn promotion" property from the
spec without inventing a second full model-training pipeline for every
individual digit specialist, which is out of scope for this build.

Promotion requires, per spec section 12 / 23 (adversarial check):
- a minimum number of scored observations
- a real rolling log-loss improvement over the champion
- the improvement to hold in BOTH halves of the evaluation window (a crude
  stability/overfitting check -- an edge that only shows up in one half is
  more likely noise or a regime blip than something the challenger should be
  promoted on)
"""
from __future__ import annotations

import statistics

from app.logging_setup import get_logger
from decision.decision_engine import SymbolPipeline
from research.experiment_log import ExperimentLog

logger = get_logger("learning.champion_challenger")


class ChampionChallengerManager:
    def __init__(self, evaluate_every_n_trades: int, min_trades_to_evaluate: int,
                 min_improvement: float, experiment_log: ExperimentLog):
        self.evaluate_every_n_trades = evaluate_every_n_trades
        self.min_trades_to_evaluate = min_trades_to_evaluate
        self.min_improvement = min_improvement
        self.experiment_log = experiment_log
        self._trades_since_eval: dict[str, int] = {}

    def on_trade_settled(self, symbol: str, pipeline: SymbolPipeline) -> None:
        count = self._trades_since_eval.get(symbol, 0) + 1
        self._trades_since_eval[symbol] = count
        if count < self.evaluate_every_n_trades:
            return
        self._trades_since_eval[symbol] = 0
        self._evaluate(symbol, pipeline)

    def _evaluate(self, symbol: str, pipeline: SymbolPipeline) -> None:
        champ = list(pipeline._ensemble_logloss_champion)
        chall = list(pipeline._ensemble_logloss_challenger)
        n = min(len(champ), len(chall))
        if n < self.min_trades_to_evaluate:
            return

        champ, chall = champ[-n:], chall[-n:]
        mid = n // 2
        # NOTE: cast every derived value to native Python float/bool
        # explicitly. `statistics.mean` on a list of numpy.float64 (which is
        # what `-np.log(...)` produces) returns numpy.float64, and comparing
        # two numpy.float64 values returns numpy.bool -- NOT a Python bool.
        # numpy.float64 happens to subclass float so it slips through JSON
        # encoding silently, but numpy.bool does not subclass bool (bool
        # can't be subclassed at all) and fails json.dumps with "Object of
        # type bool is not JSON serializable" -- which used to take down
        # every single champion/challenger evaluation log silently (`_safe`
        # swallows the exception, so nothing crashed, it just never wrote a
        # row). Caught from a live deployment log. Fixed at the source here;
        # database/repository.py also sanitizes every payload defensively so
        # this class of bug can't silently recur from anywhere else either.
        champ_loss = float(statistics.mean(champ))
        chall_loss = float(statistics.mean(chall))
        improvement = champ_loss - chall_loss  # positive == challenger better

        half1_improve = float(statistics.mean(champ[:mid]) - statistics.mean(chall[:mid])) if mid > 0 else 0.0
        half2_improve = float(statistics.mean(champ[mid:]) - statistics.mean(chall[mid:])) if (n - mid) > 0 else 0.0
        stable = bool(half1_improve > 0 and half2_improve > 0)

        promote = bool(improvement >= self.min_improvement and stable)

        self.experiment_log.record(
            symbol=symbol,
            hypothesis="performance-weighted challenger ensemble beats current champion weights",
            metrics={
                "n": n, "champion_log_loss": champ_loss, "challenger_log_loss": chall_loss,
                "improvement": improvement, "half1_improvement": half1_improve, "half2_improvement": half2_improve,
                "stable": stable,
            },
            decision="promote" if promote else "reject",
        )

        if promote:
            new_weights = pipeline.performance.current_weights()
            logger.info("Promoting challenger weights to champion", extra={"extra_fields": {
                "symbol": symbol, "improvement": improvement,
                "new_weights": {k: v.tolist() for k, v in new_weights.items()},
            }})
            pipeline.champion_weights = new_weights
