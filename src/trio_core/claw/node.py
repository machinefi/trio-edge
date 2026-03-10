"""OpenClaw Gateway WebSocket client — trio-core as a direct node.

Handles:
  - WebSocket connect/reconnect with exponential backoff
  - Challenge-response handshake with Ed25519 device identity (protocol v3)
  - Device pairing (first-time registration)
  - Authenticated connection (subsequent runs with saved token)
  - Ping/pong keepalive (30s)
  - Frame dispatch → command handlers
  - Graceful shutdown (SIGTERM/SIGINT)
  - Auth failure detection (no retry on token rejection)
  - SIGUSR1 status dump
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path

import websockets
from websockets.asyncio.client import ClientConnection

from .protocol import (
    InvokeRequest,
    InvokeResult,
    connect_params,
    generate_id,
    load_or_create_identity,
    make_req,
    make_res,
    pair_request_params,
)

logger = logging.getLogger("trio.claw")

# Constants (matching TrioClaw Go implementation)
PING_INTERVAL = 30  # seconds
MAX_BACKOFF = 15  # seconds
PAIR_TIMEOUT = 300  # 5 minutes
SHUTDOWN_TIMEOUT = 5  # seconds — forced exit if drain hangs
STATE_DIR = Path.home() / ".trio"
STATE_FILE = STATE_DIR / "claw_state.json"


class AuthError(Exception):
    """Raised when the Gateway rejects authentication (invalid/expired token)."""


class ClawNode:
    """OpenClaw node client — connects trio-core directly to the Gateway."""

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789",
        node_id: str | None = None,
        handler: object | None = None,
    ):
        self.gateway_url = gateway_url
        self.node_id = node_id or _default_node_id()
        self.handler = handler  # CommandHandler instance
        self.token = _load_token()
        self._ws: ClientConnection | None = None
        self._running = False
        self._shutdown = asyncio.Event()
        self._start_time: float = 0.0
        self._reconnect_count: int = 0
        self._connected: bool = False

        # Load or create Ed25519 device identity
        self._device_id, self._private_key, self._raw_pub = load_or_create_identity()
        logger.info("Device ID: %s", self._device_id[:16] + "...")

    # =========================================================================
    # Pairing — first-time device registration
    # =========================================================================

    async def pair(self, display_name: str = "trio-core") -> str:
        """Pair with the Gateway. Returns device token.

        The operator must approve via: openclaw devices approve <requestId>
        """
        logger.info("Connecting for pairing...")
        ws = await websockets.connect(self.gateway_url)

        try:
            # 1. Receive connect.challenge
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if frame.get("event") != "connect.challenge":
                raise RuntimeError(f"Expected connect.challenge, got: {frame}")

            nonce = _extract_nonce(frame)

            # 2. Send connect with device identity (+ gateway token if provided)
            params = connect_params(
                self.node_id, token=self.token,
                nonce=nonce,
                device_id=self._device_id,
                private_key=self._private_key,
                raw_pub=self._raw_pub,
            )
            await ws.send(json.dumps(make_req("connect", params)))

            # 3. Read hello response
            res = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            # For pairing, we may get hello-ok or need to proceed anyway

            # 4. Send pair request
            pair_params = pair_request_params(self.node_id, display_name)
            await ws.send(json.dumps(make_req("node.pair.request", pair_params)))
            logger.info("Pairing request sent. Waiting for operator approval...")
            print(f"\n  Approve in OpenClaw: openclaw devices approve\n")

            # 5. Wait for device.pair.resolved event
            loop = asyncio.get_running_loop()
            deadline = loop.time() + PAIR_TIMEOUT
            while loop.time() < deadline:
                remaining = deadline - loop.time()
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                event = json.loads(raw)
                if event.get("event") == "device.pair.resolved":
                    payload = json.loads(event.get("payloadJSON", "{}"))
                    if payload.get("status") == "approved" and payload.get("token"):
                        token = payload["token"]
                        _save_token(token)
                        self.token = token
                        logger.info("Paired successfully!")
                        return token
                    raise RuntimeError(f"Pairing {payload.get('status', 'failed')}")

            raise TimeoutError("Pairing timed out (operator did not approve)")
        finally:
            await ws.close()

    # =========================================================================
    # Main run loop — connect, dispatch, reconnect
    # =========================================================================

    async def run(self) -> None:
        """Connect to Gateway and enter main event loop. Blocks until cancelled."""
        if not self.token:
            raise RuntimeError("Not paired. Run `trio claw --pair` first.")

        self._running = True
        self._start_time = time.monotonic()
        self._shutdown.clear()
        backoff = 1.0

        # Install signal handlers
        self._install_signals()

        while self._running and not self._shutdown.is_set():
            try:
                await self._connect_and_loop()
                backoff = 1.0  # reset on clean disconnect
            except AuthError as e:
                logger.error("Authentication failed: %s", e)
                logger.error("Check your token with `trio claw --token <token>`")
                self._running = False
                break
            except (
                websockets.ConnectionClosed,
                ConnectionRefusedError,
                OSError,
                asyncio.TimeoutError,
            ) as e:
                if not self._running or self._shutdown.is_set():
                    break
                self._connected = False
                self._reconnect_count += 1
                logger.warning("Disconnected: %s. Reconnecting in %.0fs...", e, backoff)
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                    break  # shutdown requested during backoff
                except asyncio.TimeoutError:
                    pass  # normal — backoff elapsed
                backoff = min(backoff * 1.5, MAX_BACKOFF)
            except asyncio.CancelledError:
                break

        self._connected = False
        await self._close()

    async def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False
        self._shutdown.set()
        await self._close()

    def status(self) -> dict:
        """Return current node status (for SIGUSR1 / health endpoint)."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        watches = []
        if self.handler and hasattr(self.handler, "_watches"):
            for w in self.handler._watches.values():
                watches.append({
                    "watchId": w.watch_id,
                    "condition": w.condition,
                    "checks": w.checks,
                    "alerts": w.alerts,
                })
        model = "unknown"
        if self.handler and hasattr(self.handler, "engine") and self.handler.engine:
            try:
                health = self.handler.engine.health()
                model = health.get("backend", {}).get("model", "unknown")
            except Exception:
                pass
        return {
            "status": "connected" if self._connected else "reconnecting",
            "uptime_s": round(uptime, 1),
            "reconnects": self._reconnect_count,
            "watches_active": len(watches),
            "watches": watches,
            "model": model,
            "gateway": self.gateway_url,
            "node_id": self.node_id,
        }

    # =========================================================================
    # Signal handling
    # =========================================================================

    def _install_signals(self) -> None:
        """Install SIGTERM, SIGINT, and SIGUSR1 handlers."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._on_shutdown_signal, sig)

        # SIGUSR1 for status dump (Unix only)
        if hasattr(signal, "SIGUSR1"):
            loop.add_signal_handler(signal.SIGUSR1, self._on_status_signal)

    def _on_shutdown_signal(self, sig: signal.Signals) -> None:
        """Handle SIGTERM/SIGINT — initiate graceful shutdown."""
        sig_name = sig.name
        logger.info("Received %s — shutting down gracefully (timeout %ds)...",
                     sig_name, SHUTDOWN_TIMEOUT)
        self._running = False
        self._shutdown.set()

        # Schedule forced exit if graceful shutdown hangs
        loop = asyncio.get_running_loop()
        loop.call_later(SHUTDOWN_TIMEOUT, self._force_exit, sig_name)

    def _force_exit(self, sig_name: str) -> None:
        """Force exit if graceful shutdown exceeds timeout."""
        logger.warning("Graceful shutdown timed out after %ds (%s). Forcing exit.",
                       SHUTDOWN_TIMEOUT, sig_name)
        sys.exit(1)

    def _on_status_signal(self) -> None:
        """Handle SIGUSR1 — dump status to stderr."""
        s = self.status()
        lines = [
            f"--- trio-claw status ---",
            f"  status:     {s['status']}",
            f"  uptime:     {s['uptime_s']:.0f}s",
            f"  reconnects: {s['reconnects']}",
            f"  gateway:    {s['gateway']}",
            f"  model:      {s['model']}",
            f"  watches:    {s['watches_active']}",
        ]
        for w in s["watches"]:
            lines.append(f"    - {w['watchId']}: {w['condition']} "
                         f"(checks={w['checks']}, alerts={w['alerts']})")
        lines.append("---")
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()

    # =========================================================================
    # Internal: connect + event loop
    # =========================================================================

    async def _connect_and_loop(self) -> None:
        """Single connection lifecycle: handshake → event loop."""
        ws = await websockets.connect(
            self.gateway_url,
            max_size=25 * 1024 * 1024,  # 25MB max frame
        )
        self._ws = ws

        try:
            # Handshake (may raise AuthError)
            await self._handshake(ws)
            self._connected = True
            logger.info("Connected to Gateway (%s)", self.gateway_url)

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws))

            try:
                # Read loop
                async for raw in ws:
                    if self._shutdown.is_set():
                        break
                    frame = json.loads(raw)
                    frame_type = frame.get("type")

                    if frame_type == "event":
                        await self._handle_event(ws, frame)
                    elif frame_type == "req":
                        await self._handle_request(ws, frame)
                    elif frame_type == "res":
                        logger.debug("Response frame (ignored): id=%s ok=%s",
                                     frame.get("id"), frame.get("ok"))
                    else:
                        logger.debug("Unknown frame type: %s", frame_type)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
        finally:
            self._ws = None
            self._connected = False
            # Stop all watches — they can't send results without a WebSocket
            if self.handler and hasattr(self.handler, "stop_all_watches"):
                await self.handler.stop_all_watches()
            await ws.close()

    async def _handshake(self, ws: ClientConnection) -> None:
        """Challenge-response handshake with Ed25519 device identity.

        Raises AuthError if the Gateway rejects the token (don't retry).
        """
        # 1. Receive connect.challenge
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if frame.get("event") != "connect.challenge":
            raise RuntimeError(f"Expected connect.challenge, got: {frame}")

        # Extract nonce — may be in payloadJSON (double-encoded) or payload
        nonce = _extract_nonce(frame)

        # 2. Send connect with device identity + auth token
        params = connect_params(
            self.node_id, token=self.token,
            nonce=nonce,
            device_id=self._device_id,
            private_key=self._private_key,
            raw_pub=self._raw_pub,
        )
        await ws.send(json.dumps(make_req("connect", params)))

        # 3. Expect hello-ok
        res = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if res.get("type") != "res" or not res.get("ok"):
            # Distinguish auth failure from other handshake errors
            error = res.get("error", {})
            error_code = error.get("code", "") if isinstance(error, dict) else ""
            error_msg = error.get("message", str(res)) if isinstance(error, dict) else str(res)

            # Auth-related error codes from Gateway
            auth_codes = {"AUTH_FAILED", "UNAUTHORIZED", "TOKEN_EXPIRED",
                          "TOKEN_INVALID", "FORBIDDEN", "INVALID_TOKEN"}
            auth_keywords = {"auth", "token", "unauthorized", "forbidden", "credential"}

            is_auth = (
                error_code.upper() in auth_codes
                or any(kw in error_msg.lower() for kw in auth_keywords)
            )
            if is_auth:
                raise AuthError(f"Token rejected by Gateway: {error_msg}")
            raise RuntimeError(f"Handshake failed: {res}")

        payload = res.get("payload", {})
        if isinstance(payload, dict) and payload.get("type") != "hello-ok":
            raise RuntimeError(f"Expected hello-ok, got: {payload}")

    async def _handle_event(self, ws: ClientConnection, frame: dict) -> None:
        """Dispatch incoming events."""
        event = frame.get("event", "")
        logger.debug("Event received: %s", event)

        if event == "node.invoke.request":
            # Try payloadJSON first, then payload, then top-level params
            payload_json = frame.get("payloadJSON", "")
            if payload_json and payload_json != "{}":
                raw = payload_json
            elif "payload" in frame:
                p = frame["payload"]
                raw = json.dumps(p) if isinstance(p, dict) else p
            elif "params" in frame:
                raw = json.dumps(frame["params"]) if isinstance(frame["params"], dict) else frame["params"]
            else:
                raw = "{}"

            try:
                req = InvokeRequest.from_payload(raw)
            except Exception:
                logger.exception("Failed to parse invoke payload: %s", raw[:500])
                return
            logger.info("Invoke received: command=%s id=%s", req.command, req.id)
            # Dispatch asynchronously — use _safe_dispatch to log exceptions
            asyncio.create_task(self._safe_dispatch(ws, req))

        elif event == "tick":
            pass  # Gateway heartbeat, ignore

        else:
            logger.debug("Unhandled event: %s", event)

    async def _handle_request(self, ws: ClientConnection, frame: dict) -> None:
        """Handle incoming request frames (ping, invoke)."""
        method = frame.get("method", "")
        logger.debug("Request received: method=%s id=%s", method, frame.get("id"))

        if method == "ping":
            res = make_res(frame.get("id", ""), ok=True)
            await ws.send(json.dumps(res))
        elif method == "node.invoke":
            # Gateway may send invokes as req frames instead of events
            params = frame.get("params", {})
            try:
                req = InvokeRequest(
                    id=params.get("id", frame.get("id", "")),
                    node_id=params.get("nodeId", self.node_id),
                    command=params.get("command", ""),
                    params=params.get("params", {}),
                )
            except Exception:
                logger.exception("Failed to parse invoke request: %s", frame)
                return
            if not req.command:
                logger.warning("Invoke request missing command: %s", frame)
                return
            logger.info("Invoke via req: command=%s id=%s", req.command, req.id)
            asyncio.create_task(self._safe_dispatch(ws, req))
        else:
            logger.debug("Unhandled request method: %s", method)

    async def _safe_dispatch(self, ws: ClientConnection, req: InvokeRequest) -> None:
        """Wrapper that ensures dispatch exceptions are always logged."""
        try:
            await self._dispatch_invoke(ws, req)
        except Exception:
            logger.exception("Unhandled error in dispatch for %s (id=%s)", req.command, req.id)

    async def _dispatch_invoke(self, ws: ClientConnection, req: InvokeRequest) -> None:
        """Execute a command and send the result back."""
        if self.handler is None:
            result = InvokeResult(
                id=req.id, node_id=self.node_id, ok=False,
                error_code="UNAVAILABLE", error_message="No handler configured",
            )
        else:
            # Give handler access to the WebSocket for streaming results (vision.watch)
            self.handler._ws = ws
            try:
                result = await self.handler.handle(req)
            except Exception as e:
                logger.exception("Command %s failed", req.command)
                result = InvokeResult(
                    id=req.id, node_id=self.node_id, ok=False,
                    error_code="INTERNAL_ERROR", error_message=str(e),
                )

        logger.info("Invoke result: command=%s ok=%s id=%s", req.command, result.ok, req.id)
        frame = result.to_req_frame()
        try:
            payload = json.dumps(frame)
            logger.debug("Sending result frame (%d bytes)", len(payload))
            await ws.send(payload)
        except Exception:
            logger.exception("Failed to send invoke result for %s", req.id)

    async def _ping_loop(self, ws: ClientConnection) -> None:
        """Send WebSocket ping every 30 seconds."""
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.ping()
            except Exception:
                break

    async def _close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None


# =============================================================================
# Helpers
# =============================================================================

def _extract_nonce(frame: dict) -> str:
    """Extract nonce from a connect.challenge frame."""
    if "payloadJSON" in frame:
        challenge = json.loads(frame["payloadJSON"])
        nonce = challenge.get("nonce", "")
        if nonce:
            return nonce
    if "payload" in frame:
        payload = frame["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        nonce = payload.get("nonce", "")
        if nonce:
            return nonce
    return frame.get("nonce", "")


# =============================================================================
# Token persistence (~/.trio/claw_state.json)
# =============================================================================

def _load_token() -> str | None:
    """Load saved device token."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return data.get("token")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_token(token: str) -> None:
    """Save device token for subsequent connections."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"token": token}, indent=2))
    logger.info("Token saved to %s", STATE_FILE)


def _default_node_id() -> str:
    """Generate a default node ID from hostname."""
    import socket
    return f"trio-{socket.gethostname().split('.')[0].lower()}"
