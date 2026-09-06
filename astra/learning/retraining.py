from __future__ import annotations

from app.logging_setup import get_logger
from models.registry import SymbolModelRegistry

logger = get_logger("learning.retraining")


class RetrainingController:
    """Decides when Random Forest / XGBoost should be refit for a symbol.

    Two distinct rules, so that "don't retrain heavyweight models every
    tick" (spec section 11) and "learning should begin as soon as possible"
    don't fight each other:

    - FIRST fit: as soon as a batch model has buffered `min_observations`
      samples, fit it on the very next check -- don't additionally wait for
      the next `every_n`-tick cadence boundary on top of that. Before this
      fix, a model that reached 500 buffered samples at tick 501 wouldn't
      get its first fit until tick 600 (the next multiple of every_n=300),
      sitting idle and contributing nothing to the ensemble for no reason.
    - SUBSEQUENT refits: still governed by `every_n`, since re-fitting a
      Random Forest/XGBoost on literally every tick is neither necessary
      nor cheap.
    """

    def __init__(self, every_n: int, min_observations: int):
        self.every_n = every_n
        self.min_observations = min_observations
        self._since_retrain: dict[str, int] = {}

    def maybe_retrain(self, symbol: str, registry: SymbolModelRegistry) -> None:
        count = self._since_retrain.get(symbol, 0) + 1
        self._since_retrain[symbol] = count
        cadence_due = count >= self.every_n

        for name, model in registry.batch_models().items():
            if not model.ready_to_retrain(self.min_observations):
                continue
            first_fit_due = not getattr(model, "_fitted", False)
            if not (cadence_due or first_fit_due):
                continue
            try:
                model.retrain()
                logger.info("Batch model retrained", extra={"extra_fields": {
                    "symbol": symbol, "model": name, "first_fit": first_fit_due,
                }})
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch model retrain failed", exc_info=exc,
                             extra={"extra_fields": {"symbol": symbol, "model": name}})

        if cadence_due:
            self._since_retrain[symbol] = 0
