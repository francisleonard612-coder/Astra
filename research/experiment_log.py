"""
Astra's experiment registry (spec sections 24/25), scoped to what this build
actually runs: challenger-weight-vs-champion-weight evaluations, one entry
per evaluation. Each entry is self-contained enough to answer "did we try
this before, and what happened" without needing an LLM to interpret it --
per spec section 48, no LLM calls belong in the trading path, and there's no
reason a deterministic evaluation like this one needs one either.

Persists to Supabase (astra_experiment_log) when available; always keeps an
in-memory copy so the process still runs (in a degraded, unpersisted mode)
if the DB is briefly unreachable.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentRecord:
    experiment_id: int
    timestamp: float
    symbol: str
    hypothesis: str
    metrics: dict[str, Any]
    decision: str


class ExperimentLog:
    def __init__(self, repository=None):
        self._repository = repository
        self._counter = itertools.count(1)
        self._records: list[ExperimentRecord] = []

    def record(self, *, symbol: str, hypothesis: str, metrics: dict[str, Any], decision: str) -> ExperimentRecord:
        rec = ExperimentRecord(
            experiment_id=next(self._counter),
            timestamp=time.time(),
            symbol=symbol,
            hypothesis=hypothesis,
            metrics=metrics,
            decision=decision,
        )
        self._records.append(rec)
        if self._repository is not None:
            try:
                self._repository.insert_experiment(rec)
            except Exception:  # noqa: BLE001
                pass
        return rec

    def history_for(self, symbol: str) -> list[ExperimentRecord]:
        return [r for r in self._records if r.symbol == symbol]
