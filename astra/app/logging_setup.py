"""
Structured logging for Astra.

Per the current build scope, the monitoring dashboard/alerting subsystem
(spec section 30) is intentionally OUT of scope -- Railway's log viewer plus
the Supabase tables (astra_predictions, astra_trades, astra_regime_log,
astra_system_events) are the source of truth for now. This module just makes
sure every log line is structured JSON so it's easy to grep/parse later, and
gives every component its own named logger.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Merges per-call `extra={"extra_fields": {...}}` with the adapter's
    own baseline fields, instead of the stdlib LoggerAdapter.process(),
    which does `kwargs["extra"] = self.extra` -- unconditionally
    OVERWRITING (not merging) whatever `extra` was passed at the call site.

    Every logger in this codebase is created via get_logger(name) with no
    baseline kwargs, so self.extra was always {}, and the stdlib behavior
    silently discarded every single per-call extra_fields dict app-wide --
    symbol, contract_type, the actual exception string, etc. all vanished
    before reaching JsonFormatter. That's why "Proposal request failed"
    warnings were logged with no symbol or error message: the informative
    part of the call was thrown away by the adapter, not by the formatter.
    """
    def process(self, msg, kwargs):
        call_extra = kwargs.get("extra") or {}
        call_fields = call_extra.get("extra_fields") or {}
        base_fields = (self.extra or {}).get("extra_fields") or {}
        merged = {**base_fields, **call_fields}
        kwargs["extra"] = {"extra_fields": merged} if merged else {}
        return msg, kwargs


def get_logger(name: str, **extra_fields) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    return ContextLoggerAdapter(logger, {"extra_fields": extra_fields} if extra_fields else {})
