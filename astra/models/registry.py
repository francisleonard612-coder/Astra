from __future__ import annotations

from dataclasses import dataclass, field

from models.base import DigitModel
from models.bayesian import BayesianModel
from models.frequency import EWMAFrequencyModel, RollingFrequencyModel, UniformModel
from models.markov import MarkovModel
from models.sklearn_models import LogisticModel, RandomForestModel, XGBoostModel


def build_model_set(max_markov_order: int = 3, rolling_window: int = 500) -> dict[str, DigitModel]:
    """A fresh set of model instances for one symbol. Models carry per-symbol
    learned state (SGD weights, RF/XGB fits, EWMA vectors) so each symbol
    needs its own instances -- these are cheap to construct."""
    models: dict[str, DigitModel] = {
        "uniform": UniformModel(),
        "rolling_frequency": RollingFrequencyModel(window=rolling_window),
        "ewma_frequency": EWMAFrequencyModel(),
        "bayesian": BayesianModel(),
        "markov": MarkovModel(max_order=max_markov_order),
        "logistic": LogisticModel(),
        "random_forest": RandomForestModel(),
        "xgboost": XGBoostModel(),
    }
    return models


@dataclass
class ModelRegistryEntry:
    model_id: str
    model_type: str
    symbol: str
    status: str = "active"  # experimental | challenger | champion | deprecated | rejected
    version: int = 1
    creation_timestamp: str | None = None
    promotion_timestamp: str | None = None
    performance_metrics: dict = field(default_factory=dict)


class SymbolModelRegistry:
    """Tracks model instances plus their registry metadata for one symbol."""

    def __init__(self, symbol: str, max_markov_order: int, rolling_window: int):
        self.symbol = symbol
        self.models: dict[str, DigitModel] = build_model_set(max_markov_order, rolling_window)
        self.entries: dict[str, ModelRegistryEntry] = {
            name: ModelRegistryEntry(model_id=f"{symbol}:{name}:v1", model_type=name, symbol=symbol)
            for name in self.models
        }

    def batch_models(self) -> dict[str, DigitModel]:
        return {n: m for n, m in self.models.items() if n in ("random_forest", "xgboost")}
