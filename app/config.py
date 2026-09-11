"""
Central configuration for Astra.

Loads configs/config.yaml as the base and lets a small set of environment
variables override the values that matter most for a given deployment
(credentials, stakes, risk limits). Everything else is edited in the YAML
file directly -- that keeps the research parameters (windows, thresholds,
ensemble weights) in one readable place instead of scattered across env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


@dataclass
class DerivConfig:
    app_id: str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    # Legacy WS host, only used as a last-resort fallback if Deriv's OTP response
    # ever omits the ready-to-use websocket URL (see DerivClient._exchange_otp).
    ws_url: str = field(default_factory=lambda: os.getenv("DERIV_WS_URL", "wss://api.derivws.com/trading/v1/options/ws/demo"))
    # REST API base URL. DerivClient appends /trading/v1/options/accounts[...]
    # itself, so this should just be the host -- do not include a path.
    options_token_url: str = field(
        default_factory=lambda: os.getenv("DERIV_OPTIONS_TOKEN_URL", "https://api.derivws.com")
    )
    # Optional: pin a specific Options account id (e.g. "DOT90004580") instead of
    # letting DerivClient auto-resolve one from GET /accounts. Leave unset to
    # auto-resolve based on use_real_account below.
    account_id: str | None = field(default_factory=lambda: os.getenv("DERIV_ACCOUNT_ID", "").strip() or None)
    # Which account type to auto-resolve to when account_id is unset.
    # False (default) = demo. True = real money. This is independent of
    # DRY_RUN (see AstraConfig.dry_run below) -- the two combine as:
    #   use_real=False, dry_run=True  -> nothing hits Deriv's buy endpoint;
    #                                     safest option, good for a first
    #                                     smoke test of signals/connectivity.
    #   use_real=False, dry_run=False -> trades actually execute through
    #                                     Deriv's real proposal/buy/settlement
    #                                     path, but against a demo account's
    #                                     play balance. This is "paper trading
    #                                     that exercises the real order flow",
    #                                     not a local simulation -- the
    #                                     recommended way to validate the bot
    #                                     end-to-end before risking money.
    #   use_real=True,  dry_run=False -> live, real-money trading.
    #   use_real=True,  dry_run=True  -> connects to the real account but
    #                                     never places an order; mostly
    #                                     useful for checking real-account
    #                                     balance/symbol access without risk.
    use_real_account: bool = field(default_factory=lambda: _env_bool("DERIV_USE_REAL", False))


@dataclass
class SupabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    service_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", ""))
    enabled: bool = field(default_factory=lambda: _env_bool("SUPABASE_ENABLED", True))


@dataclass
class RiskOverrides:
    base_stake: float = field(default_factory=lambda: _env_float("BASE_STAKE", 1.0))
    max_stake: float = field(default_factory=lambda: _env_float("MAX_STAKE", 5.0))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 5))
    max_daily_loss: float = field(default_factory=lambda: _env_float("MAX_DAILY_LOSS", 25.0))
    max_trades_per_day: int = field(default_factory=lambda: _env_int("MAX_TRADES_PER_DAY", 500))
    max_concurrent_trades: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_TRADES", 2))
    martingale_enabled: bool = field(default_factory=lambda: _env_bool("MARTINGALE_ENABLED", False))
    martingale_factor: float = field(default_factory=lambda: _env_float("MARTINGALE_FACTOR", 2.0))
    martingale_max_steps: int = field(default_factory=lambda: _env_int("MARTINGALE_MAX_STEPS", 3))


class AstraConfig:
    """Loads YAML config and applies env overrides. Access via `cfg = AstraConfig()`."""

    def __init__(self, path: str | None = None):
        path = path or os.getenv("CONFIG_PATH", "configs/config.yaml")
        p = Path(path)
        if not p.is_absolute():
            # resolve relative to project root (this file lives in app/)
            p = Path(__file__).resolve().parent.parent / path
        with open(p, "r") as f:
            self.raw: dict[str, Any] = yaml.safe_load(f)

        self.deriv = DerivConfig()
        self.supabase = SupabaseConfig()
        self.risk_overrides = RiskOverrides()

        # apply env overrides onto the risk section of raw so downstream code
        # that reads cfg.raw["risk"] sees a single consistent view
        self.raw.setdefault("risk", {})
        self.raw["risk"]["base_stake"] = self.risk_overrides.base_stake
        self.raw["risk"]["max_stake"] = self.risk_overrides.max_stake
        self.raw["risk"]["max_consecutive_losses"] = self.risk_overrides.max_consecutive_losses
        self.raw["risk"]["max_daily_loss"] = self.risk_overrides.max_daily_loss
        self.raw["risk"]["max_trades_per_day"] = self.risk_overrides.max_trades_per_day
        self.raw["risk"]["max_concurrent_trades"] = self.risk_overrides.max_concurrent_trades
        self.raw["risk"].setdefault("staking", {})
        self.raw["risk"]["staking"]["enabled"] = self.risk_overrides.martingale_enabled
        self.raw["risk"]["staking"]["progression_factor"] = self.risk_overrides.martingale_factor
        self.raw["risk"]["staking"]["max_steps"] = self.risk_overrides.martingale_max_steps

        # See DerivConfig.use_real_account docstring above for how this
        # combines with account type to give four distinct modes.
        self.dry_run: bool = _env_bool("DRY_RUN", False)
        self.currency: str = os.getenv("CURRENCY", "USD")
        self.log_level: str = os.getenv("LOG_LEVEL", self.raw.get("logging", {}).get("level", "INFO"))

        symbols_override = os.getenv("ASTRA_SYMBOLS", "").strip()
        self.symbol_override: list[str] | None = (
            [s.strip() for s in symbols_override.split(",") if s.strip()] if symbols_override else None
        )

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


_CONFIG: AstraConfig | None = None


def get_config() -> AstraConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AstraConfig()
    return _CONFIG
