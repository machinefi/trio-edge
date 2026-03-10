"""OpenClaw Gateway WebSocket protocol (v3) — wire types.

Protocol overview:
  - Transport: WebSocket on port 18789
  - Frame types: req (request), res (response), event
  - All frames are JSON text messages
  - payloadJSON fields are DOUBLE-ENCODED: a JSON string containing JSON

Connection flow:
  1. WS connect
  2. Receive: event "connect.challenge" with nonce
  3. Send: req "connect" with caps, commands, auth token
  4. Receive: res with "hello-ok"
  5. Ready for invocations

Reference: TrioClaw Go implementation (github.com/machinefi/TrioClaw)
"""

from __future__ import annotations

import json
import platform
import random
import string
from dataclasses import dataclass, field


# =============================================================================
# Wire frame types
# =============================================================================

def make_req(method: str, params: dict, req_id: str | None = None) -> dict:
    """Build a request frame."""
    return {
        "type": "req",
        "id": req_id or generate_id(),
        "method": method,
        "params": params,
    }


def make_res(req_id: str, ok: bool, payload: dict | None = None,
             error: dict | None = None) -> dict:
    """Build a response frame."""
    frame = {"type": "res", "id": req_id, "ok": ok}
    if payload:
        frame["payload"] = payload
    if error:
        frame["error"] = error
    return frame


def make_event(event: str, payload: dict | None = None) -> dict:
    """Build an event frame."""
    frame = {"type": "event", "event": event}
    if payload is not None:
        frame["payloadJSON"] = json.dumps(payload)
    return frame


# =============================================================================
# Connect params
# =============================================================================

def connect_params(
    node_id: str,
    token: str | None = None,
    caps: list[str] | None = None,
    commands: list[str] | None = None,
    version: str = "0.1.0",
) -> dict:
    """Build connect request params."""
    if caps is None:
        caps = ["camera", "vision"]
    if commands is None:
        commands = [
            "camera.snap",
            "camera.list",
            "vision.analyze",
            "vision.watch",
            "vision.watch.stop",
            "vision.status",
        ]

    params = {
        "minProtocol": 3,
        "maxProtocol": 3,
        "client": {
            "id": "trio-core",
            "version": version,
            "platform": platform.system().lower(),
            "deviceFamily": "trio-core",
            "modelIdentifier": node_id,
            "mode": "node",
        },
        "role": "node",
        "caps": caps,
        "commands": commands,
    }
    if token:
        params["auth"] = {"token": token}
    return params


def pair_request_params(
    node_id: str,
    display_name: str,
    caps: list[str] | None = None,
    commands: list[str] | None = None,
    version: str = "0.1.0",
) -> dict:
    """Build pair request params."""
    if caps is None:
        caps = ["camera", "vision"]
    if commands is None:
        commands = [
            "camera.snap", "camera.list",
            "vision.analyze", "vision.watch",
            "vision.watch.stop", "vision.status",
        ]

    return {
        "nodeId": node_id,
        "displayName": display_name,
        "platform": platform.system().lower(),
        "version": version,
        "deviceFamily": "trio-core",
        "caps": caps,
        "commands": commands,
        "silent": False,
    }


# =============================================================================
# Invoke types
# =============================================================================

@dataclass
class InvokeRequest:
    """Parsed invoke request from gateway."""
    id: str
    node_id: str
    command: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload_json: str) -> InvokeRequest:
        data = json.loads(payload_json)
        # params may be double-encoded
        params = data.get("params", {})
        if isinstance(params, str):
            params = json.loads(params)
        return cls(
            id=data["id"],
            node_id=data.get("nodeId", ""),
            command=data["command"],
            params=params,
        )


@dataclass
class InvokeResult:
    """Result to send back to gateway after executing a command."""
    id: str
    node_id: str
    ok: bool
    payload: dict | None = None
    error_code: str | None = None
    error_message: str | None = None

    def to_req_frame(self) -> dict:
        """Build a req frame with method 'node.invoke.result'."""
        result: dict = {
            "id": self.id,
            "nodeId": self.node_id,
            "ok": self.ok,
        }
        if self.ok and self.payload is not None:
            result["payloadJSON"] = json.dumps(self.payload)
        if not self.ok and self.error_code:
            result["error"] = {
                "code": self.error_code,
                "message": self.error_message or "",
            }
        return make_req("node.invoke.result", result)


# =============================================================================
# Utilities
# =============================================================================

def generate_id(length: int = 8) -> str:
    """Generate a random request ID."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def decode_payload_json(payload_json: str) -> dict:
    """Decode a double-encoded payloadJSON string."""
    return json.loads(payload_json)
