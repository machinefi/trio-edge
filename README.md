<p align="center">
  <h1 align="center">Trio Core</h1>
  <p align="center">
    <strong>Open-source camera intelligence for your network</strong>
  </p>
  <p align="center">
    ONVIF discovery + RTSP streaming + YOLO detection + VLM scene understanding.<br>
    One pip install. Runs on any Mac, Linux, or Windows machine.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/trio-core/"><img src="https://img.shields.io/pypi/v/trio-core?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/trio-core/"><img src="https://img.shields.io/pypi/pyversions/trio-core" alt="Python"></a>
  <a href="https://github.com/machinefi/trio-core/blob/main/LICENSE"><img src="https://img.shields.io/github/license/machinefi/trio-core" alt="License"></a>
  <a href="https://github.com/machinefi/trio-core/stargazers"><img src="https://img.shields.io/github/stars/machinefi/trio-core?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#two-modes">Two Modes</a> |
  <a href="#cloud-relay">Cloud Relay</a> |
  <a href="#local-ai">Local AI</a> |
  <a href="#cli">CLI</a> |
  <a href="#api-reference">API</a> |
  <a href="#python-sdk">SDK</a> |
  <a href="#architecture">Architecture</a>
</p>

---

## What is Trio Core?

Trio Core is the open-source camera agent for [Trio AI](https://trio.ai). It runs on your local network, discovers cameras via ONVIF, and either:

1. **Relays video to Trio Cloud** for cloud ingest, analysis, and dashboards
2. **Runs AI locally** for live monitoring, counting, and scene understanding on your own hardware

```
┌──────────────────────────────────────────────────────────┐
│  Your Network                                            │
│                                                          │
│  IP Camera ──RTSP──► Trio Core ──HTTPS──► Trio Cloud     │
│                      (this repo)         (paid, $99/cam) │
│                          │                               │
│                          └── or run AI locally           │
│                             (free, open source)          │
└──────────────────────────────────────────────────────────┘
```

**Core capabilities:**
- **Discover** — Auto-find cameras on your network via ONVIF and resolve RTSP URLs
- **Monitor** — Live RTSP camera analysis with watch prompts, object counts, and event digests
- **Relay** — Stream RTSP, webcam, or video-file sources to Trio Cloud over HTTP MPEG-TS
- **Analyze** — Run scene understanding on images and videos from the CLI or API
- **Serve** — Expose local inference APIs for detection and description
- **Tailscale auto-proxy** — Works through Tailscale networks automatically

---

## Quick Start

```bash
# If Python is managed by Homebrew and pip reports an "externally managed environment",
# create a venv first:
# python3 -m venv .venv && source .venv/bin/activate
pip install 'trio-core[mlx]'      # Apple Silicon local AI
pip install 'trio-core[cuda]'     # NVIDIA GPU local AI
pip install trio-core             # Discovery, relay, API

# Check your setup
trio doctor

# Discover cameras on your network
trio discover

# Start local monitoring (Apple Silicon default)
trio cam --source rtsp://admin:pass@192.168.1.100/stream

# Or relay to Trio Cloud
trio relay --source rtsp://admin:pass@192.168.1.100/stream \
           --token YOUR_TOKEN
```

---

## Two Modes

### Mode 1: Cloud Relay (Trio Cloud customers)

Trio Core takes an RTSP stream, webcam, or video file, registers a camera with Trio Cloud, and pushes HTTP MPEG-TS to the cloud ingest endpoint. All AI processing happens in the cloud.

```bash
trio relay --source rtsp://admin:pass@192.168.1.100/stream \
           --token YOUR_TOKEN
```

**What Cloud does:** session management → ingest → analysis → dashboard

### Mode 2: Local AI (open-source users)

Run everything locally on your own machine. No cloud needed, no subscription.

```bash
# Default local monitor (Apple Silicon)
trio cam --source rtsp://admin:pass@192.168.1.100/stream

# Count objects
trio cam --source rtsp://... --count

# Smart event digest
trio cam --source rtsp://... --digest

# Analyze a saved image or video
trio analyze photo.jpg -q "What's here?"
```

**Note:** Use `--model` and `--backend` to override the default local model selection. Local mode gives you real-time detection and descriptions, but no persistent memory, entity tracking, historical analytics, or dashboard. For those features, use [Trio Cloud](https://trio.ai).

---

## Features

### ONVIF Camera Discovery

```bash
trio discover
# Found 2 camera(s):
#   [1] Reolink RLC-810A (192.168.1.100)
#       RTSP: rtsp://192.168.1.100:554/h264Preview_01_main
#   [2] Hikvision DS-2CD2143 (192.168.1.101)
#       RTSP: rtsp://192.168.1.101:554/Streaming/Channels/101
```

### Tailscale Auto-Proxy

If you use Tailscale, Trio Core automatically detects when the macOS network extension blocks camera access and creates a transparent TCP proxy:

```
trio cam --source rtsp://admin:pass@192.168.1.100/stream
# Tailscale detected — starting TCP proxy via system Python...
# Proxy: 127.0.0.1:15554 → 192.168.1.100:554
# (continues normally, user sees no difference)
```

### YOLO Object Detection

Built-in YOLOv10n (ONNX, 9MB) with tiled detection for accuracy:

```bash
trio cam --source rtsp://... --count
# [14:23:46] People: 3, Vehicles: 2
# [14:24:12] People: 5, Vehicles: 2 (+2 people)
```

### VLM Scene Description

Supports multiple local AI configurations:

| Mode | Command | Notes |
|------|---------|-------|
| Default local monitor | `trio cam --source rtsp://...` | Uses the built-in default model for live monitoring |
| Custom local model | `trio cam --source rtsp://... --model <MODEL_ID>` | Override the Hugging Face model ID |
| Transformers backend | `trio analyze photo.jpg --backend transformers --model Qwen/Qwen2.5-VL-3B-Instruct` | CUDA or CPU |
| Adapter / fine-tune | `trio cam --source rtsp://... --adapter ./adapter_dir` | Load a LoRA adapter directory |

---

## CLI

```bash
trio discover                                 # Find cameras via ONVIF
trio cam --source rtsp://... --count            # Live monitor + object counts
trio cam --host 192.168.1.100 -p pass         # Resolve RTSP via ONVIF + monitor
trio cam --source 0 -w "person at the door"   # Local webcam monitor with alerts
trio cam --source rtsp://... --digest           # Event timeline with scene understanding
trio relay --source rtsp://... --token ...    # Relay to Trio Cloud
trio relay --discover -p pass --token ...     # Discover a camera and relay it
trio serve                                    # Start inference API server
trio analyze photo.jpg -q "What's here?"      # Analyze a single image or video
trio bench video.mp4                          # Benchmark inference speed
trio doctor                                   # System check, hardware info, model rec
trio claw --camera rtsp://... --gateway ws://127.0.0.1:18789  # OpenClaw node
```

---

## API Reference

Start the local inference server:

```bash
trio serve                          # default: 0.0.0.0:8100
trio serve --port 9000              # custom port
TRIO_API_KEY=secret trio serve      # enable auth
```

### `POST /api/inference/detect`

YOLO object detection.

```bash
curl -X POST http://localhost:8100/api/inference/detect \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "'$(base64 -i photo.jpg)'"}'
```

```json
{
  "people_count": 3, "vehicle_count": 1,
  "by_class": {"person": 3, "car": 1},
  "crops_b64": [{"class": "person", "bbox": [100, 50, 200, 300], "confidence": 0.92}],
  "elapsed_ms": 45
}
```

### `POST /api/inference/describe`

VLM scene description.

```bash
curl -X POST http://localhost:8100/api/inference/describe \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "'$(base64 -i photo.jpg)'", "prompt": "Describe what you see."}'
```

### `POST /api/inference/crop-describe`

Combined: YOLO detects → crop → VLM describes each entity → full scene description.

---

## Python SDK

```python
from trio_core import TrioCore, EngineConfig

engine = TrioCore()
engine.load()

result = engine.analyze_video("photo.jpg", "What do you see?")
print(result.text)
```

---

## Supported Models

### Default models (auto-selected by hardware)

| Hardware | Model | Quantization | VRAM |
|---|---|---|---|
| Apple Silicon (≥32GB) | Qwen3-VL-8B | MLX 4-bit | ~5GB |
| Apple Silicon (<32GB) | Qwen3.5-2B | MLX 4-bit | ~2GB |
| NVIDIA GPU (≥16GB) | Qwen3-VL-8B | AWQ 4-bit | ~5GB |
| NVIDIA GPU (<16GB) | Qwen3.5-2B | AWQ 4-bit | ~2GB |

> No GPU? trio-core requires Apple Silicon or NVIDIA GPU. Run `trio doctor` to check.

### Tier 2 — Inference only (via mlx-vlm)

Gemma 3n, SmolVLM2, Phi-4, FastVLM, and any model supported by mlx-vlm.

---

## Architecture

```
                          Trio Core
                              |
              +---------------+---------------+
              |                               |
         YOLO Pipeline                   VLM Pipeline
              |                               |
    YOLOv10n ONNX (9MB)            Qwen/Claude/GPT/any LLM
    tiled 2x2 detection              native MLX loading
    ByteTrack tracking               ToMe token compression
              |                       KV cache reuse
              |                               |
              +---------------+---------------+
                              |
              +-------+-------+-------+
              |       |       |       |
          /detect  /describe  /crop   Relay
                              -describe  to Cloud
```

### Trio Cloud integration

When connected to Trio Cloud, Edge is just a lightweight relay:

```
Camera → Edge (RTSP pull + compress) → Trio Cloud (all AI in cloud)
```

Edge sends ~50-100 KB/s per camera. No GPU needed on the edge device.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TRIO_MODEL` | `Qwen3-VL-8B-4bit` | HuggingFace model ID |
| `TRIO_YOLO_MODEL` | (auto-downloaded) | Path to YOLO ONNX model |
| `TRIO_API_KEY` | (none) | Bearer token for API auth |
| `TRIO_CLOUD_URL` | (none) | Trio Cloud API URL for relay mode |
| `TRIO_CLOUD_TOKEN` | (none) | Trio Cloud auth token |

See [`src/trio_core/config.py`](src/trio_core/config.py) for all options.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `trio discover` finds no cameras | Make sure cameras are on the same subnet. Some routers block multicast. |
| Camera found but can't connect | Check username/password. Try `trio cam --source rtsp://admin:pass@IP/stream` directly. |
| Tailscale blocking camera access | Trio Core auto-detects this and creates a proxy. If it doesn't work, try `trio doctor`. |
| First run slow | Model download (~2-5 GB). Subsequent runs start instantly. |
| Out of memory | Use a smaller model: `TRIO_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit` |

Run `trio doctor` to diagnose most issues.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
