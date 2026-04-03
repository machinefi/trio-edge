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

1. **Relays frames to Trio Cloud** for full AI analysis (memory, entity tracking, dashboards)
2. **Runs AI locally** with your own LLM (Claude, GPT, local Qwen) for standalone use

```
┌──────────────────────────────────────────────────────────┐
│  Your Network                                            │
│                                                          │
│  IP Camera ──RTSP──► Trio Core ──HTTPS──► Trio Cloud     │
│                      (this repo)         (paid, $99/cam) │
│                          │                               │
│                          └── or use your own LLM         │
│                             (free, open source)          │
└──────────────────────────────────────────────────────────┘
```

**Core capabilities:**
- **Discover** — Auto-find cameras on your network via ONVIF
- **Relay** — Push RTSP frames to Trio Cloud over HTTPS (NAT-friendly)
- **Detect** — YOLO v10n object detection (people, vehicles, 80 classes)
- **Describe** — VLM scene descriptions with any LLM (local or cloud)
- **Tailscale auto-proxy** — Works through Tailscale networks automatically

---

## Quick Start

```bash
# Install
pip install 'trio-core[mlx]'      # Apple Silicon
pip install 'trio-core[cuda]'     # NVIDIA GPU
pip install trio-core              # CPU-only

# Discover cameras on your network
trio discover

# Start watching a camera
trio cam --rtsp rtsp://admin:pass@192.168.1.100/stream

# Or relay to Trio Cloud
trio relay --camera rtsp://admin:pass@192.168.1.100/stream \
           --cloud https://api.trio.ai --token YOUR_TOKEN
```

---

## Two Modes

### Mode 1: Cloud Relay (Trio Cloud customers)

Trio Core pulls RTSP or local video, registers a camera with Trio Cloud, and pushes HTTP MPEG-TS to the cloud ingest endpoint. All AI processing happens in the cloud.

```bash
trio relay --camera rtsp://admin:pass@192.168.1.100/stream \
           --cloud https://api.trio.ai \
           --token YOUR_TOKEN
```

**What Edge does:** source capture → ffmpeg MPEG-TS mux → authenticated HTTP upload
**What Cloud does:** session management → ingest → analysis → dashboard

### Mode 2: Local AI (open-source users)

Run everything locally with your own LLM. No cloud needed, no subscription.

```bash
# With local Qwen model (Apple Silicon)
trio cam --rtsp rtsp://admin:pass@192.168.1.100/stream

# With Claude API
ANTHROPIC_API_KEY=sk-xxx trio cam --rtsp rtsp://... --llm claude

# With OpenAI
OPENAI_API_KEY=sk-xxx trio cam --rtsp rtsp://... --llm gpt-4o
```

**Note:** Local mode gives you real-time detection and descriptions, but no persistent memory, entity tracking, historical analytics, or dashboard. For those features, use [Trio Cloud](https://trio.ai).

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
trio cam --rtsp rtsp://admin:pass@192.168.1.100/stream
# Tailscale detected — starting TCP proxy via system Python...
# Proxy: 127.0.0.1:15554 → 192.168.1.100:554
# (continues normally, user sees no difference)
```

### YOLO Object Detection

Built-in YOLOv10n (ONNX, 9MB) with tiled detection for accuracy:

```bash
trio cam --rtsp rtsp://... --count
# [14:23:46] People: 3, Vehicles: 2
# [14:24:12] People: 5, Vehicles: 2 (+2 people)
```

### VLM Scene Description

Supports multiple VLM backends:

| Backend | Command | Requirements |
|---------|---------|-------------|
| Local Qwen (MLX) | `trio cam --rtsp ...` | Apple Silicon, 4GB+ RAM |
| Claude | `trio cam --llm claude` | `ANTHROPIC_API_KEY` |
| GPT-4o | `trio cam --llm gpt-4o` | `OPENAI_API_KEY` |
| Any OpenAI-compatible | `trio cam --llm-url http://...` | API endpoint |

---

## CLI

```bash
trio discover                          # Find cameras via ONVIF
trio cam --rtsp rtsp://... --count     # Watch + count objects
trio cam --host 192.168.1.100 -p pass  # Auto-discover + connect
trio relay --camera rtsp://... --cloud https://api.trio.ai  # Cloud relay
trio serve                             # Start inference API server
trio analyze photo.jpg -q "What's here?"  # Analyze single image
trio webcam -w "person at the door"    # Webcam with alerts
trio doctor                            # Diagnose setup issues
trio device                            # Show hardware info
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

### Tier 1 — Full optimization (native loading + visual token compression + KV reuse)

| Model | Params | 4-bit VRAM | Best for |
|---|---|---|---|
| Qwen3-VL-8B | 8B | ~5GB | **Recommended** — best accuracy |
| Qwen2.5-VL-3B | 3B | ~2GB | Fast, lightweight |
| Qwen3.5 | 0.8-9B | 0.5-5G | Flexible range |
| InternVL3 | 1-2B | 1-1.6G | Tiny devices |

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
| Camera found but can't connect | Check username/password. Try `trio cam --rtsp rtsp://admin:pass@IP/stream` directly. |
| Tailscale blocking camera access | Trio Core auto-detects this and creates a proxy. If it doesn't work, try `trio doctor`. |
| First run slow | Model download (~2-5 GB). Subsequent runs start instantly. |
| Out of memory | Use a smaller model: `TRIO_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit` |

Run `trio doctor` to diagnose most issues.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
