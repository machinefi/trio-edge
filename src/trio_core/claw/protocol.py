"""OpenClaw Gateway WebSocket protocol (v3) — wire types + device identity.

Protocol overview:
  - Transport: WebSocket on port 18789
  - Frame types: req (request), res (response), event
  - All frames are JSON text messages
  - payloadJSON fields are DOUBLE-ENCODED: a JSON string containing JSON
  - Device identity: Ed25519 keypair, SHA-256 device ID, signed auth payload

Connection flow:
  1. WS connect
  2. Receive: event "connect.challenge" with nonce
  3. Send: req "connect" with caps, commands, device identity (signed nonce)
  4. Receive: res with "hello-ok"
  5. Ready for invocations
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path


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
# Device identity — Ed25519 keypair + SHA-256 device ID
# =============================================================================

STATE_DIR = Path.home() / ".trio"
IDENTITY_FILE = STATE_DIR / "device_identity.json"


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    """Base64url decode (handles missing padding)."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def load_or_create_identity() -> tuple:
    """Load or generate Ed25519 identity. Returns (device_id, private_key, raw_pub)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    if IDENTITY_FILE.exists():
        data = json.loads(IDENTITY_FILE.read_text())
        priv_bytes = _b64url_decode(data["privateKey"])
        private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
        raw_pub = _b64url_decode(data["publicKey"])
        device_id = data["deviceId"]
        return device_id, private_key, raw_pub

    # Generate new keypair
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    raw_pub = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    raw_priv = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    device_id = hashlib.sha256(raw_pub).hexdigest()

    # Save
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    IDENTITY_FILE.write_text(json.dumps({
        "deviceId": device_id,
        "publicKey": _b64url_encode(raw_pub),
        "privateKey": _b64url_encode(raw_priv),
    }, indent=2))

    return device_id, private_key, raw_pub


def build_device_params(
    nonce: str,
    device_id: str,
    private_key: object,
    raw_pub: bytes,
    token: str | None = None,
    client_id: str = "node-host",
    client_mode: str = "node",
    role: str = "node",
    plat: str | None = None,
    device_family: str = "trio-core",
) -> dict:
    """Build the `device` object for connect params with Ed25519 signature."""
    signed_at_ms = int(time.time() * 1000)
    plat = (plat or platform.system()).strip().lower()
    device_family = device_family.strip().lower()

    # v3 payload: pipe-delimited string
    payload = "|".join([
        "v3",
        device_id,
        client_id,
        client_mode,
        role,
        "",  # scopes (empty for nodes)
        str(signed_at_ms),
        token or "",
        nonce,
        plat,
        device_family,
    ])

    signature = private_key.sign(payload.encode("utf-8"))

    return {
        "id": device_id,
        "publicKey": _b64url_encode(raw_pub),
        "signature": _b64url_encode(signature),
        "signedAt": signed_at_ms,
        "nonce": nonce,
    }


# =============================================================================
# Connect params
# =============================================================================

def connect_params(
    node_id: str,
    token: str | None = None,
    nonce: str | None = None,
    device_id: str | None = None,
    private_key: object | None = None,
    raw_pub: bytes | None = None,
    caps: list[str] | None = None,
    commands: list[str] | None = None,
    version: str = "0.1.0",
) -> dict:
    """Build connect request params with device identity."""
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
            "id": "node-host",
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

    # Add device identity if we have the key material
    if device_id and private_key and raw_pub:
        params["device"] = build_device_params(
            nonce=nonce,
            device_id=device_id,
            private_key=private_key,
            raw_pub=raw_pub,
            token=token,
        )

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
