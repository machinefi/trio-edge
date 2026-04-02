# ONVIF Discovery Script Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `uv run` Python script that discovers ONVIF cameras on the local network and prints normalized ONVIF endpoint information in text or JSON form.

**Architecture:** The implementation will live in a single standalone script under `scripts/` with a small local dataclass and narrowly scoped functions for probe sending, response collection, response parsing, normalization, and output formatting. A dedicated pytest module will load the script as a module and drive it through pure-function tests plus a thin fake-socket integration seam so malformed payloads, duplicate responses, and empty scans are covered without requiring real cameras.

**Tech Stack:** Python 3.12+, `argparse`, `socket`, `dataclasses`, `json`, `urllib.parse`, `pytest`, `uv`

---

## Chunk 1: Standalone Discovery Script

### Task 1: Add the test harness for script loading and parse behavior

**Files:**
- Create: `tests/test_discover_onvif_script.py`
- Create: `scripts/discover_onvif.py`
- Reference: `docs/superpowers/specs/2026-04-02-onvif-discovery-script-design.md`

- [ ] **Step 1: Write the failing test for module loading and valid response parsing**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "discover_onvif.py"
    spec = importlib.util.spec_from_file_location("discover_onvif_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_parse_response_extracts_name_ip_port_and_onvif_url():
    module = load_script_module()
    payload = (
        b"<d:XAddrs>http://192.168.1.10:8000/onvif/device_service</d:XAddrs> "
        b"onvif://www.onvif.org/name/Front%20Door"
    )

    camera = module.parse_response(payload, "192.168.1.10")

    assert camera is not None
    assert camera.name == "Front Door"
    assert camera.ip == "192.168.1.10"
    assert camera.port == 8000
    assert camera.onvif_url == "http://192.168.1.10:8000/onvif/device_service"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_discover_onvif_script.py::test_parse_response_extracts_name_ip_port_and_onvif_url -v`
Expected: FAIL because `scripts/discover_onvif.py` does not exist yet

- [ ] **Step 3: Write the minimal script skeleton and parser implementation**

```python
from dataclasses import dataclass
import re
from urllib.parse import urlsplit, urlunsplit


@dataclass(slots=True)
class DiscoveredCamera:
    name: str
    ip: str
    port: int
    onvif_url: str


def normalize_xaddr(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    include_port = port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    )
    netloc = hostname if not include_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, "", ""))


def extract_name(text: str, source_ip: str) -> str:
    match = re.search(r"onvif://www\\.onvif\\.org/name/([^\\s<]+)", text)
    if not match:
        return f"Camera @ {source_ip}"
    return match.group(1).replace("%20", " ")


def parse_response(payload: bytes, source_ip: str) -> DiscoveredCamera | None:
    text = payload.decode("utf-8", errors="replace")
    match = re.search(r"https?://[^\\s<\\\"]+", text)
    if not match:
        return None
    onvif_url = normalize_xaddr(match.group(0))
    parsed = urlsplit(onvif_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return DiscoveredCamera(
        name=extract_name(text, source_ip),
        ip=source_ip,
        port=port,
        onvif_url=onvif_url,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_discover_onvif_script.py::test_parse_response_extracts_name_ip_port_and_onvif_url -v`
Expected: PASS

- [ ] **Step 5: Commit the first TDD slice**

```bash
git add tests/test_discover_onvif_script.py scripts/discover_onvif.py
git commit -m "test: add ONVIF discovery script parse coverage"
```

### Task 2: Add normalization, fallback naming, malformed-response skipping, and formatting

**Files:**
- Modify: `tests/test_discover_onvif_script.py`
- Modify: `scripts/discover_onvif.py`

- [ ] **Step 1: Write failing tests for normalization, fallback naming, malformed payloads, and output formatting**

```python
def test_parse_response_uses_fallback_name_when_scope_name_missing():
    module = load_script_module()

    camera = module.parse_response(
        b"<d:XAddrs>http://192.168.1.11/onvif/device_service</d:XAddrs>",
        "192.168.1.11",
    )

    assert camera is not None
    assert camera.name == "Camera @ 192.168.1.11"


def test_parse_response_returns_none_for_payload_without_xaddr():
    module = load_script_module()

    assert module.parse_response(b"<xml>broken</xml>", "192.168.1.12") is None


def test_normalize_xaddr_drops_default_port_and_query():
    module = load_script_module()

    assert (
        module.normalize_xaddr("HTTP://192.168.1.13:80/onvif/device_service?wsdl#frag")
        == "http://192.168.1.13/onvif/device_service"
    )


def test_format_text_uses_stable_camera_blocks():
    module = load_script_module()
    cameras = [
        module.DiscoveredCamera(
            name="B",
            ip="192.168.1.20",
            port=80,
            onvif_url="http://192.168.1.20/onvif/device_service",
        ),
        module.DiscoveredCamera(
            name="A",
            ip="192.168.1.10",
            port=8080,
            onvif_url="http://192.168.1.10:8080/onvif/device_service",
        ),
    ]

    assert module.format_text(cameras) == (
        "Name: A\\n"
        "IP: 192.168.1.10\\n"
        "Port: 8080\\n"
        "ONVIF URL: http://192.168.1.10:8080/onvif/device_service\\n\\n"
        "Name: B\\n"
        "IP: 192.168.1.20\\n"
        "Port: 80\\n"
        "ONVIF URL: http://192.168.1.20/onvif/device_service"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_discover_onvif_script.py -k "fallback or malformed or normalize or format_text" -v`
Expected: FAIL because the helper functions and formatting behavior are not complete yet

- [ ] **Step 3: Implement the minimal formatting helpers**

```python
def format_text(cameras: list[DiscoveredCamera]) -> str:
    if not cameras:
        return "No ONVIF cameras discovered."
    blocks = []
    for camera in sorted(cameras, key=lambda item: (item.ip, item.onvif_url)):
        blocks.append(
            "\\n".join(
                [
                    f"Name: {camera.name}",
                    f"IP: {camera.ip}",
                    f"Port: {camera.port}",
                    f"ONVIF URL: {camera.onvif_url}",
                ]
            )
        )
    return "\\n\\n".join(blocks)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_discover_onvif_script.py -k "fallback or malformed or normalize or format_text" -v`
Expected: PASS

- [ ] **Step 5: Commit the helper and formatting slice**

```bash
git add tests/test_discover_onvif_script.py scripts/discover_onvif.py
git commit -m "feat: normalize and format standalone ONVIF discovery output"
```

### Task 3: Add socket collection, deduplication, JSON output, and CLI entrypoint

**Files:**
- Modify: `tests/test_discover_onvif_script.py`
- Modify: `scripts/discover_onvif.py`

- [ ] **Step 1: Write failing tests for deduplication, empty JSON output, timeout validation, and fake-socket discovery**

```python
import json
import pytest


class FakeSocket:
    def __init__(self, responses):
        self.responses = list(responses)

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendto(self, payload, addr):
        self.sent = (payload, addr)

    def recvfrom(self, _size):
        if self.responses:
            return self.responses.pop(0)
        raise socket.timeout

    def close(self):
        self.closed = True


def test_discover_cameras_deduplicates_equivalent_xaddrs(monkeypatch):
    module = load_script_module()
    fake_socket = FakeSocket(
        [
            (
                b"<d:XAddrs>http://192.168.1.30:80/onvif/device_service</d:XAddrs>",
                ("192.168.1.30", 3702),
            ),
            (
                b"<d:XAddrs>http://192.168.1.30/onvif/device_service?wsdl</d:XAddrs>",
                ("192.168.1.30", 3702),
            ),
        ]
    )
    monkeypatch.setattr(module, "build_socket", lambda: fake_socket)

    cameras = module.discover_cameras(timeout=0.1)

    assert len(cameras) == 1


def test_collect_responses_stops_on_socket_timeout():
    module = load_script_module()
    fake_socket = FakeSocket([])

    assert module.collect_responses(fake_socket, timeout=0.1) == []


def test_main_returns_exit_code_1_on_fatal_socket_error(monkeypatch, capsys):
    module = load_script_module()

    def fail_build_socket():
        raise OSError("network down")

    monkeypatch.setattr(module, "build_socket", fail_build_socket)

    exit_code = module.main(["--timeout", "0.1"])

    assert exit_code == 1
    assert "network down" in capsys.readouterr().err


def test_format_json_returns_empty_array_for_no_cameras():
    module = load_script_module()

    assert json.loads(module.format_json([])) == []


def test_format_json_returns_camera_objects():
    module = load_script_module()

    cameras = [
        module.DiscoveredCamera(
            name="Front Door",
            ip="192.168.1.31",
            port=80,
            onvif_url="http://192.168.1.31/onvif/device_service",
        )
    ]

    assert json.loads(module.format_json(cameras)) == [
        {
            "name": "Front Door",
            "ip": "192.168.1.31",
            "port": 80,
            "onvif_url": "http://192.168.1.31/onvif/device_service",
        }
    ]


def test_parse_timeout_rejects_non_positive_values():
    module = load_script_module()

    with pytest.raises(SystemExit):
        module.main(["--timeout", "0"])
    with pytest.raises(SystemExit):
        module.main(["--timeout", "-1"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_discover_onvif_script.py -k "deduplicates or format_json or timeout" -v`
Expected: FAIL because socket orchestration, JSON formatting, and CLI validation are not implemented yet

- [ ] **Step 3: Implement the minimal socket and CLI behavior**

```python
PROBE_PAYLOAD = b"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:discover-onvif-script</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>
  </e:Body>
</e:Envelope>"""


def build_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    return sock


def send_probe(sock: socket.socket) -> None:
    sock.sendto(PROBE_PAYLOAD, ("239.255.255.250", 3702))


def collect_responses(sock: socket.socket, timeout: float) -> list[tuple[bytes, str]]:
    sock.settimeout(timeout)
    packets = []
    while True:
        try:
            payload, (source_ip, _source_port) = sock.recvfrom(8192)
        except socket.timeout:
            return packets
        packets.append((payload, source_ip))


def discover_cameras(timeout: float) -> list[DiscoveredCamera]:
    sock = build_socket()
    try:
        send_probe(sock)
        seen = set()
        cameras = []
        for payload, source_ip in collect_responses(sock, timeout):
            camera = parse_response(payload, source_ip)
            if camera is None:
                continue
            key = (camera.ip, camera.onvif_url)
            if key in seen:
                continue
            seen.add(key)
            cameras.append(camera)
        return sorted(cameras, key=lambda item: (item.ip, item.onvif_url))
    finally:
        sock.close()


def format_json(cameras: list[DiscoveredCamera]) -> str:
    return json.dumps(
        [
            {
                "name": camera.name,
                "ip": camera.ip,
                "port": camera.port,
                "onvif_url": camera.onvif_url,
            }
            for camera in cameras
        ]
    )


def positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than 0")
    return timeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=positive_timeout, default=3.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        cameras = discover_cameras(timeout=args.timeout)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_json(cameras) if args.json else format_text(cameras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full focused test file and verify it passes**

Run: `uv run pytest tests/test_discover_onvif_script.py -v`
Expected: PASS

- [ ] **Step 5: Run the script itself for a local smoke test**

Run: `uv run python scripts/discover_onvif.py --json --timeout 0.5`
Expected: Either `[]` or a JSON array of discovered cameras. No traceback.

- [ ] **Step 6: Commit the complete standalone discovery script**

```bash
git add tests/test_discover_onvif_script.py scripts/discover_onvif.py
git commit -m "feat: add standalone ONVIF discovery script"
```

### Task 4: Final verification and handoff

**Files:**
- Modify: `scripts/discover_onvif.py` only if verification exposes a real bug
- Modify: `tests/test_discover_onvif_script.py` only if verification exposes a real bug

- [ ] **Step 1: Run the focused verification commands**

Run: `uv run pytest tests/test_discover_onvif_script.py -v`
Expected: PASS

Run: `uv run pytest tests/test_onvif.py tests/test_cli_onvif.py -v`
Expected: PASS

- [ ] **Step 2: Re-run the standalone script in text mode**

Run: `uv run python scripts/discover_onvif.py --timeout 0.5`
Expected: Either a stable camera block listing or `No ONVIF cameras discovered.`

- [ ] **Step 3: Commit any verification-driven fixes**

```bash
git add scripts/discover_onvif.py tests/test_discover_onvif_script.py
git commit -m "test: finalize standalone ONVIF discovery verification"
```
