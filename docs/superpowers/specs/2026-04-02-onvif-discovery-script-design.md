# ONVIF Discovery Script Design

**Date:** 2026-04-02

**Goal:** Add a small standalone Python script that discovers ONVIF cameras on the local network and prints discovered cameras plus their ONVIF endpoints, runnable with `uv run`.

## Scope

This design covers one standalone script under `scripts/` for local network discovery only.

In scope:
- Send a WS-Discovery probe to the ONVIF multicast address.
- Collect responses for a configurable timeout.
- Parse camera identity and ONVIF XAddr endpoint data from responses.
- Print discovered cameras in human-readable form.
- Support machine-readable JSON output for later CLI comparison.
- Add focused tests for parsing, deduplication, and empty-result behavior.

Out of scope:
- RTSP URI resolution.
- Authentication or credential handling.
- Changes to the existing `trio` or `trio-edge` CLI.
- Refactoring the shared `trio_core.onvif` module unless a concrete, independently justified bug is found later.

## Requirements

The script must:
- Run as a standalone local tool via `uv run python scripts/discover_onvif.py`.
- Prefer its own discovery and parsing flow rather than depending on the existing shared ONVIF implementation for correctness.
- Use standard-library networking and parsing where practical.
- Tolerate malformed responses by skipping them instead of aborting the scan.
- Exit successfully when no cameras are found and print a clear no-results message.
- Surface real socket or multicast setup failures as user-visible errors with a nonzero exit code.
- Use a concrete default discovery timeout of `3` seconds.

## Approaches Considered

### 1. Standalone script with its own WS-Discovery implementation

Write a self-contained script that builds the probe payload, sends it over UDP multicast, parses responses, deduplicates results, and formats output.

Pros:
- Isolates the script from suspected bugs in the shared ONVIF module.
- Easier to reason about and debug independently.
- Provides a cleaner baseline before later CLI integration.

Cons:
- Duplicates some logic that already exists in the library.

### 2. Thin wrapper around `trio_core.onvif.discover_cameras()`

Pros:
- Fastest to implement.
- Reuses existing code and behavior.

Cons:
- Inherits discovery and parsing bugs from the shared module.
- Harder to tell whether failures come from the script or library.

### 3. Depend on an external discovery package or shell tool

Pros:
- Less code in-repo.

Cons:
- Adds dependency and portability risk for a simple local scan.
- Reduces control over parsing and output shape.

## Recommended Approach

Use approach 1. The script will own the network probe, response parsing, deduplication, and output formatting. It may import small helper utilities from the repo later if they are clearly correct and helpful, but correctness should not depend on the existing `trio_core.onvif` discovery path.

## Script Design

### File

- Create `scripts/discover_onvif.py`

### Responsibilities

The script will contain:
- A probe payload builder or constant for WS-Discovery.
- A `send_probe()` function that sends the WS-Discovery `Probe` envelope to multicast address `239.255.255.250:3702` over UDP.
- A `collect_responses()` function that receives datagrams until the configured timeout expires.
- A `parse_response(payload: bytes, source_ip: str)` function that extracts:
  - response IP from the UDP sender tuple, not the payload
  - ONVIF XAddr URL
  - port
  - best-effort camera name from ONVIF scopes
- A `discover_cameras()` coordinator that parses valid responses and deduplicates results by `(ip, onvif_url)` so duplicate responses from the same endpoint do not create duplicate output rows.
- A `format_text()` function for human-readable output.
- A `format_json()` function for machine-readable output.
- A small `main()` entrypoint using `argparse`.

The probe envelope should be a minimal WS-Discovery request for `dn:NetworkVideoTransmitter`, matching ONVIF camera discovery expectations without adding authentication or vendor-specific fields.

### Inputs

Supported flags:
- `--timeout <seconds>`: receive window for discovery responses. Default: `3`.
- `--json`: emit JSON instead of human-readable output.

### Outputs

Human-readable output will print one block per discovered camera with:
- `name`
- `ip`
- `port`
- `onvif_url`

The text format should be deterministic and stable:

```text
Name: <name>
IP: <ip>
Port: <port>
ONVIF URL: <onvif_url>
```

Each camera block will be separated by a single blank line. Cameras will be sorted by `ip` and then `onvif_url` before formatting so repeated runs produce stable output ordering when the network returns the same results.

If no cameras are found, print a clear message such as `No ONVIF cameras discovered.` and exit with status `0`.

JSON output will emit a list of objects with the same fields. This is intended to make later comparison against CLI behavior straightforward.

## Parsing and Data Model

The script should use a small local dataclass, for example `DiscoveredCamera`, with:
- `name: str`
- `ip: str`
- `port: int`
- `onvif_url: str`

Parsing rules:
- Extract XAddr candidates from each response payload and use the first normalized HTTP or HTTPS XAddr.
- Derive `port` from the XAddr when present; otherwise fall back to `80`.
- Prefer ONVIF scope names from `/name/`.
- If no explicit name is present, fall back to `Camera @ <ip>`.
- Ignore malformed payloads that do not expose a usable HTTP or HTTPS XAddr endpoint.
- Ignore payloads that do not provide enough information to construct a valid `DiscoveredCamera`.

Normalization rules:
- Lowercase the XAddr scheme and hostname.
- Remove an explicit default port of `80` for `http` and `443` for `https` when constructing the deduplication key.
- Preserve the path, but treat an empty path as `/`.
- Drop query string and fragment components for deduplication.
- Preserve the original normalized XAddr string in output.

## Error Handling

Expected handling:
- Malformed response payload: skip response and continue scanning.
- Duplicate device response: ignore duplicate and keep the first normalized result.
- No devices discovered: successful exit with empty result indication.
- Socket creation, multicast send, or receive setup failure: print the error and exit nonzero.

Socket contract:
- Create an IPv4 UDP socket.
- Bind it to `("", 0)` so the operating system selects an ephemeral local port for replies.
- Set the receive timeout to the configured timeout window.
- Treat `socket.timeout` during receive as the normal end-of-scan condition once no more responses arrive.
- Treat other `OSError` failures during setup, send, or receive as fatal scan errors.

The script should not silently swallow setup failures. Network-level failures matter because they indicate the scan did not actually run correctly.

## Testing Strategy

Add focused tests that exercise the standalone script behavior without requiring real cameras:
- Response parsing with a valid XAddr and ONVIF name scope.
- Deduplication across repeated responses from the same device.
- Fallback naming when no scope name exists.
- Empty scan result behavior.
- Malformed response skipping.
- Socket or multicast setup failure handling.
- JSON output shape.

Networking code should stay thin so tests can patch socket reads and feed synthetic payloads into `collect_responses()` and `parse_response()` independently.

## Integration Notes

This script is intentionally standalone and separate from the existing CLI. If the script behaves correctly in local testing, its behavior can later be compared against and then folded into the main CLI without changing the original goal here.

## Non-Goals

The script will not:
- authenticate to cameras
- resolve RTSP URLs
- open or verify streams
- alter existing ONVIF library behavior as part of this task
