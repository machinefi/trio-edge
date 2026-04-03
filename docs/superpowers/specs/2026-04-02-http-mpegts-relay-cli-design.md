# HTTP MPEG-TS Relay CLI Design

Date: 2026-04-02

## Summary

Replace the `trio-edge` WHIP/WebRTC relay client with an HTTP MPEG-TS ingest client.

The scope in `trio-edge` is centered on the CLI:

- keep `trio relay` as the command name
- remove WHIP/WebRTC-specific internals and naming
- explicitly create or register a camera in `trio-cortex`
- stream ffmpeg output as `video/mp2t` to `POST /api/stream/ingest/{camera_id}`

No `api` package changes are required in `trio-edge`. The major code changes are in `src/trio_core/cli.py` and the transport module it uses.

## Goals

- Remove WHIP/WebRTC transport logic from `trio-edge`
- Make HTTP MPEG-TS ingest the only cloud relay transport
- Keep `trio relay` source-agnostic for RTSP, ONVIF-resolved RTSP, webcam, and file inputs
- Support deterministic synthetic camera IDs for non-camera sources to avoid cross-user collisions
- Keep the implementation and user-facing behavior concentrated in the CLI path

## Non-Goals

- Adding any new `api` layer inside `trio-edge`
- Preserving WHIP compatibility or fallback transport
- Changing `trio-cortex` server design in this spec
- Adding complex reconnect orchestration in the first pass
- Reworking unrelated local camera analysis commands such as `trio cam`

## Current Context

Today `trio-edge` cloud relay uses:

- `trio relay` in `src/trio_core/cli.py`
- `src/trio_core/whip_relay.py`
- tests in `tests/test_whip_relay.py`

That path depends on aiortc, SDP negotiation, peer connection lifecycle, and H.264 Annex B handling tailored to WHIP/WebRTC.

The target server contract in `trio-cortex` has already moved to:

- explicit camera creation via `POST /api/cameras`
- camera-scoped ingest via `POST /api/stream/ingest/{camera_id}`
- raw MPEG-TS upload with `Content-Type: video/mp2t`

This means `trio-edge` no longer needs browser-style real-time media negotiation. It needs a CLI-managed registration step plus a long-lived HTTP upload.

## Proposed Architecture

### 1. CLI-Centered Relay Flow

`src/trio_core/cli.py` remains the primary entry point for cloud relay.

`trio relay` keeps its role as the user-facing command, but its semantics become HTTP-ingest based rather than WHIP-based. The command should no longer describe its destination as a WHIP endpoint. Instead, it should accept the cloud base URL and token needed to:

1. register the camera with `trio-cortex`
2. derive the ingest URL for that camera
3. start a long-lived MPEG-TS upload

Source selection remains unchanged in principle:

- RTSP URL directly
- ONVIF-discovered or host-resolved RTSP URL
- webcam device
- local video file

### 2. Transport Module Replacement

`src/trio_core/whip_relay.py` should be removed and replaced by a dedicated HTTP ingest transport module.

Its responsibilities are:

- derive or accept a camera ID
- register the camera through the existing server API
- build the ffmpeg pipeline for the selected source
- stream ffmpeg stdout to the ingest endpoint as `video/mp2t`
- translate registration, ffmpeg, and upload failures into CLI-facing relay errors

It should not contain:

- aiortc
- SDP
- peer connection lifecycle
- RTP packet handling
- WHIP-specific authentication or session teardown

### 3. Deterministic Camera Identity

The relay flow needs a stable camera ID even when the source is not a real network camera.

Rules:

- If the user passes an explicit camera ID, use it.
- Otherwise derive one deterministically.
- For generic sources such as webcam or file, generate a synthetic UUID from:
  - host MAC address
  - normalized source fingerprint or stream path hash

This gives the same source on the same machine a stable ID while reducing cross-user collisions on the server.

The generated ID should be deterministic and formatted as a UUID string so it is easy to store, log, and reuse.

### 4. Explicit Camera Registration

Before starting ingest, `trio-edge` explicitly calls the server camera-creation API.

The registration payload should include:

- the derived camera ID
- a human-readable name
- the source URL or source descriptor appropriate for the server record
- metadata indicating the source is edge-managed and uses HTTP MPEG-TS ingest

The relay flow should treat camera registration as a normal prerequisite step, not as implicit behavior inside ingest upload.

## Data Flow

1. User runs `trio relay` with a source, cloud URL, and token.
2. CLI resolves the actual source:
   - direct RTSP URL
   - ONVIF/host-resolved RTSP
   - webcam device
   - file path
3. CLI determines the camera ID:
   - explicit user-provided ID, or
   - deterministic derived ID
4. CLI calls `POST /api/cameras` to create or register the camera.
5. CLI constructs the ingest URL `POST /api/stream/ingest/{camera_id}`.
6. CLI starts ffmpeg:
   - RTSP sources use copy where possible
   - webcam and file sources encode to H.264 and mux into MPEG-TS
7. CLI opens one authenticated HTTP request with `Content-Type: video/mp2t`.
8. ffmpeg stdout is streamed directly into the request body.
9. On clean shutdown, CLI stops ffmpeg and closes the upload.
10. On registration or transport failure, CLI exits with a relay error.

## FFmpeg Expectations

The transport should emit MPEG-TS instead of WHIP-oriented H.264 packet streams.

Expected direction:

- RTSP inputs:
  - prefer `-c:v copy` when feasible
  - mux to `-f mpegts`
- webcam inputs:
  - encode with low-latency H.264 settings
  - mux to `-f mpegts`
- file inputs:
  - re-encode or rate-shape as needed for continuous relay behavior
  - mux to `-f mpegts`

The first implementation should optimize for correctness and operational simplicity rather than over-tuning per-source transcoding.

## Error Handling

The relay command should fail early and clearly for:

- invalid CLI argument combinations
- invalid resolution format
- missing `ffmpeg`
- failure to derive or normalize a source
- camera registration rejection
- unauthorized or forbidden cloud requests
- ingest endpoint rejection
- ffmpeg startup failure
- ffmpeg exiting unexpectedly during upload
- mid-stream HTTP upload failure

Error messages should describe HTTP ingest or cloud ingest, never WHIP or WebRTC.

Synthetic identity generation should have a fallback if MAC address lookup fails. Relay should not become unusable solely because the primary machine identifier is unavailable.

## Testing

Replace WHIP-oriented tests with HTTP-ingest relay tests.

Coverage should include:

- `trio relay --help` output and argument validation
- deterministic camera ID generation
- registration request construction
- ffmpeg command generation for RTSP, webcam, and file sources
- HTTP ingest request setup with bearer auth and `video/mp2t`
- expected failure handling for registration errors
- expected failure handling for ingest upload errors

The old WHIP tests should be removed or rewritten once the transport is replaced.

## File Impact

Primary changes:

- `src/trio_core/cli.py`
- new HTTP ingest transport module replacing `src/trio_core/whip_relay.py`
- `tests/test_whip_relay.py` replaced with tests for the new relay flow
- relay-related README/help text updated to remove WHIP language

No `trio-edge` `api` package changes are part of this design.

## Open Design Decisions Resolved

- Major changes stay centered on `src/trio_core/cli.py`
- No new `api` layer is added in `trio-edge`
- WHIP internals are removed entirely
- Camera bootstrap is explicit, not implicit at ingest time
- Non-camera sources use deterministic synthetic UUIDs derived from host identity plus source fingerprint

## Success Criteria

- `trio relay` no longer depends on aiortc or WHIP/WebRTC flow
- users can relay RTSP, webcam, and file sources through the new HTTP ingest path
- non-camera sources register stable synthetic camera IDs
- the CLI performs explicit camera registration before ingest
- tests validate the new registration and upload path
