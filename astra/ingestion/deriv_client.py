"""
Deriv API client for Astra.

Two hard lessons from earlier bots in this account are baked in here:

1. AUTH: use Deriv's REST OTP Options API token exchange to get a
   pre-authenticated WebSocket URL, instead of the legacy
   connect-then-send-authorize-message flow. The legacy flow has produced
   401s at handshake time on this account before; the OTP exchange sidesteps
   that entirely.

2. NO INLINE AWAITS IN THE RECV LOOP: `_recv_pump` is the only coroutine that
   reads off the socket. If it ever `await`s a tick handler directly, and that
   handler calls something like `proposal()` or `buy()` which needs to read a
   response off the *same* socket, you get a self-deadlock -- the handler
   waits forever for a message that only `_recv_pump` can deliver, but
   `_recv_pump` is blocked awaiting the handler. This previously caused 100%
   of trade attempts to fail with "no response" / 1011 keepalive timeouts.
   The fix: `_recv_pump` never awaits handlers. Tick messages are pushed onto
   a per-symbol `asyncio.Queue` and consumed by independent worker tasks;
   request/response calls (proposal, buy, active_symbols, ticks_history) are
   resolved via a dict of `asyncio.Future`s keyed by req_id, and
   `_recv_pump` only ever does `future.set_result(...)`, never `await`.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

import httpx
import websockets

from app.logging_setup import get_logger

logger = get_logger("ingestion.deriv_client")


class DerivAuthError(RuntimeError):
    pass


class DerivRequestError(RuntimeError):
    def __init__(self, message: str, code: str | None = None, raw: dict | None = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}


@dataclass
class Tick:
    symbol: str
    epoch: int
    quote: float
    digit: int


class DerivClient:
    def __init__(self, app_id: str, api_token: str, ws_url: str, options_token_url: str,
                 request_timeout: float = 15.0):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = ws_url
        self.options_token_url = options_token_url
        self.request_timeout = request_timeout

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id_counter = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_queues: dict[str, asyncio.Queue] = {}
        self._tick_workers: dict[str, asyncio.Task] = {}
        self._subscription_ids: dict[str, str] = {}  # symbol -> deriv subscription id
        self._recv_task: asyncio.Task | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        async with self._connect_lock:
            auth_url = await self._exchange_otp()
            self._ws = await websockets.connect(auth_url, ping_interval=20, ping_timeout=20, close_timeout=5)
            self._closed = False
            self._recv_task = asyncio.create_task(self._recv_pump(), name="deriv-recv-pump")
            logger.info("Connected to Deriv", extra={"extra_fields": {"event_type": "ws_connected"}})

    async def _exchange_otp(self) -> str:
        """Exchange the long-lived API token for a pre-authenticated WS URL."""
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        params = {"app_id": self.app_id, "api_token": self.api_token}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.options_token_url, json=params)
        if resp.status_code != 200:
            raise DerivAuthError(f"OTP token exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
        data = resp.json()
        auth_url = data.get("websocket_url") or data.get("ws_url") or data.get("url")
        if not auth_url:
            # Fall back: some deployments return the OTP itself rather than a full URL.
            otp = data.get("otp") or data.get("token")
            if not otp:
                raise DerivAuthError(f"OTP exchange response had no usable URL/token: {data}")
            auth_url = f"{self.ws_url}?app_id={self.app_id}&otp={otp}"
        return auth_url

    async def close(self) -> None:
        self._closed = True
        for task in list(self._tick_workers.values()):
            task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def ensure_connected(self) -> None:
        if self._ws is None or self._ws.closed:
            logger.warning("Reconnecting to Deriv", extra={"extra_fields": {"event_type": "ws_reconnect"}})
            await self.connect()
            # re-subscribe any symbols we were watching
            for symbol in list(self._subscription_ids.keys()):
                self._subscription_ids.pop(symbol, None)
                await self._resubscribe(symbol)

    async def _resubscribe(self, symbol: str) -> None:
        queue = self._tick_queues.get(symbol)
        if queue is not None:
            await self._send_subscribe_request(symbol)

    # ------------------------------------------------------------------ #
    # Low-level request/response
    # ------------------------------------------------------------------ #
    async def _send(self, payload: dict) -> dict:
        """Send a request and await its matching response. Never called from _recv_pump."""
        await self.ensure_connected()
        req_id = next(self._req_id_counter)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
            result = await asyncio.wait_for(fut, timeout=self.request_timeout)
        finally:
            self._pending.pop(req_id, None)
        if "error" in result:
            err = result["error"]
            raise DerivRequestError(err.get("message", "Deriv API error"), code=err.get("code"), raw=result)
        return result

    async def _recv_pump(self) -> None:
        """The ONLY coroutine allowed to read from the socket. Never awaits handlers."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error("Malformed message from Deriv", extra={"extra_fields": {"raw": raw[:200]}})
                    continue

                msg_type = msg.get("msg_type")
                req_id = msg.get("req_id")

                if msg_type == "tick" and req_id is None:
                    self._route_tick(msg)
                    continue

                if req_id is not None and req_id in self._pending:
                    fut = self._pending[req_id]
                    if not fut.done():
                        fut.set_result(msg)
                    # a "tick" response also carries the FIRST tick + a
                    # subscription id -- route that first tick too.
                    if msg_type == "tick":
                        self._route_tick(msg)
                    continue

                # Unmatched message (e.g. late subscription tick after we
                # stopped waiting) -- route ticks, log everything else.
                if msg_type == "tick":
                    self._route_tick(msg)
                else:
                    logger.debug("Unrouted message", extra={"extra_fields": {"msg_type": msg_type}})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Recv pump crashed", exc_info=exc,
                         extra={"extra_fields": {"event_type": "recv_pump_error"}})
            # fail every pending future so callers don't hang forever
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)

    def _route_tick(self, msg: dict) -> None:
        tick = msg.get("tick")
        if not tick:
            return
        symbol = tick.get("symbol")
        queue = self._tick_queues.get(symbol)
        if queue is None:
            return
        quote = float(tick["quote"])
        digit = _last_digit(quote, tick.get("pip_size"))
        parsed = Tick(symbol=symbol, epoch=int(tick["epoch"]), quote=quote, digit=digit)
        try:
            queue.put_nowait(parsed)
        except asyncio.QueueFull:
            logger.warning("Tick queue full, dropping tick",
                            extra={"extra_fields": {"symbol": symbol, "event_type": "queue_overflow"}})
        if "id" in tick:
            self._subscription_ids[symbol] = tick["id"]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def get_active_synthetic_symbols(self, prefixes: list[str]) -> list[str]:
        resp = await self._send({"active_symbols": "brief", "product_type": "basic"})
        symbols = []
        for s in resp.get("active_symbols", []):
            if s.get("market") != "synthetic_index":
                continue
            code = s.get("symbol", "")
            if any(code.startswith(p) for p in prefixes):
                symbols.append(code)
        return sorted(set(symbols))

    async def subscribe_ticks(self, symbol: str, queue_size: int = 2000) -> asyncio.Queue:
        if symbol not in self._tick_queues:
            self._tick_queues[symbol] = asyncio.Queue(maxsize=queue_size)
        await self._send_subscribe_request(symbol)
        return self._tick_queues[symbol]

    async def _send_subscribe_request(self, symbol: str) -> None:
        resp = await self._send({"ticks": symbol, "subscribe": 1})
        sub = resp.get("subscription", {})
        if sub.get("id"):
            self._subscription_ids[symbol] = sub["id"]

    async def get_history(self, symbol: str, count: int = 5000) -> list[Tick]:
        resp = await self._send({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks",
        })
        history = resp.get("history", {})
        prices = history.get("prices", [])
        times = history.get("times", [])
        out = []
        for t, p in zip(times, prices):
            p = float(p)
            out.append(Tick(symbol=symbol, epoch=int(t), quote=p, digit=_last_digit(p, None)))
        return out

    async def get_proposal(self, symbol: str, contract_type: str, barrier: int, stake: float,
                            duration: int, duration_unit: str, currency: str) -> dict:
        resp = await self._send({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            "symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
            "barrier": str(barrier),
        })
        return resp.get("proposal", {})

    async def buy(self, proposal_id: str, price: float) -> dict:
        resp = await self._send({"buy": proposal_id, "price": price})
        return resp.get("buy", {})

    async def get_balance(self) -> dict:
        resp = await self._send({"balance": 1})
        return resp.get("balance", {})

    async def wait_for_contract_settlement(self, contract_id: int, timeout: float = 30.0) -> dict:
        resp = await self._send({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
        })
        contract = resp.get("proposal_open_contract", {})
        deadline = time.monotonic() + timeout
        while not contract.get("is_sold") and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
            contract = resp.get("proposal_open_contract", {})
        return contract


def _last_digit(quote: float, pip_size: int | float | None) -> int:
    """Extract the last significant digit of a Deriv quote given its pip size."""
    if pip_size:
        decimals = len(str(pip_size).split(".")[-1]) if "." in str(pip_size) else 0
    else:
        s = f"{quote}"
        decimals = len(s.split(".")[-1]) if "." in s else 0
    scaled = round(quote * (10 ** decimals))
    return int(scaled % 10)
