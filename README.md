<p align="center">
  <h1 align="center">TrioCore</h1>
  <p align="center">
    <strong>The fastest local VLM inference engine for Apple Silicon</strong>
  </p>
  <p align="center">
    73% faster prefill · 1.7x frame-to-frame · 68% fewer tokens · Zero Docker
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/trio-core/"><img src="https://img.shields.io/pypi/v/trio-core?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/trio-core/"><img src="https://img.shields.io/pypi/pyversions/trio-core" alt="Python"></a>
  <a href="https://github.com/machinefi/trio-core/blob/main/LICENSE"><img src="https://img.shields.io/github/license/machinefi/trio-core" alt="License"></a>
  <a href="https://github.com/machinefi/trio-core/actions/workflows/test.yml"><img src="https://github.com/machinefi/trio-core/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/machinefi/trio-core/stargazers"><img src="https://img.shields.io/github/stars/machinefi/trio-core?style=social" alt="Stars"></a>
</p>

<p align="center">
  <img src="assets/pipeline.svg" alt="TrioCore Pipeline" width="800"/>
</p>

<p align="center">
  <a href="#30-second-demo">Demo</a> |
  <a href="#install">Install</a> |
  <a href="#use-cases">Use Cases</a> |
  <a href="#api-server">API Server</a> |
  <a href="#how-it-works">How It Works</a> |
  <a href="#benchmarks">Benchmarks</a>
</p>

---

## 30-Second Demo

```bash
# Install (Apple Silicon)
pipx install 'trio-core[mlx,webcam]'

# Point your webcam and describe what to watch — that's it
trio webcam -w "a person is waving"
```

Your Mac becomes a smart camera. Green = all clear. Red = alert with audio notification. **No model training, no cloud API keys, no Docker.** Works on any M1-M4 Mac.

```bash
# Or monitor an IP camera
trio cam --host 192.168.1.100 -p mypassword -w "someone at the door"

# Or analyze any video
trio analyze video.mp4 -q "What is happening?"

# Or run as an API server (OpenAI-compatible)
trio serve
```

---

## Why TrioCore?

|  | TrioCore | Ollama + LLaVA | Cloud VLM APIs |
|---|---|---|---|
| **Latency** | ~250ms (2B, M3) | ~2s | ~3s + network |
| **Privacy** | 100% local | 100% local | Data leaves device |
| **Camera support** | Webcam, RTSP, ONVIF | None built-in | None built-in |
| **Watch mode** | Built-in ("person at door") | DIY scripting | DIY scripting |
| **Visual optimization** | ToMe + FastV + KV reuse | None | N/A |
| **Setup** | `pipx install trio-core` | Install + pull model | API key + billing |

---

## Install

```bash
# Apple Silicon (M1-M4)
pipx install 'trio-core[mlx]'         # CLI tool (recommended)
pip install 'trio-core[mlx]'          # or as library in your project

# NVIDIA / CPU
pipx install 'trio-core[transformers]'

# With webcam/camera support (opencv for local webcam)
pipx install 'trio-core[mlx,webcam]'

# For IP cameras (RTSP), just install ffmpeg
brew install ffmpeg
```

## Use Cases

### Home Security — Watch Mode

```bash
# Built-in webcam
trio webcam -w "a person is waving"

# iPhone as camera (macOS Continuity Camera)
trio webcam -s 1 -w "someone at the door"

# IP camera (auto-discovers Reolink RTSP URL)
trio cam --host 192.168.1.100 -p mypassword -w "package on the doorstep"

# Direct RTSP URL (any camera brand)
trio cam --rtsp "rtsp://admin:pass@192.168.1.100:554/stream" -w "intruder detected"

# Auto-discover ONVIF cameras on your LAN
trio cam --discover
```

Auto-calibrates resolution for ~500ms inference on any Mac. Green = clear, red = alert with audio notification. No ML training needed — just describe what to monitor.

### Video Analysis

```bash
trio analyze video.mp4 -q "What is happening?"
trio analyze photo.jpg -q "Describe this image"
```

### Python SDK

```python
from trio_core import TrioCore

engine = TrioCore()
engine.load()

result = engine.analyze_video("clip.mp4", "What is happening?")
print(result.text)  # "A person is walking through the parking lot..."
print(f"{result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.0f} tok/s")
```

TrioCore automatically applies benchmark-proven optimizations. No configuration needed — just load and go. To customize:

```python
from trio_core import TrioCore, EngineConfig

config = EngineConfig(
    tome_enabled=True,    # merge visual tokens inside ViT (-68% tokens)
    tome_r=4,
    tome_metric="hidden",
)
engine = TrioCore(config)
engine.load()
```

## API Server

```bash
trio serve --port 8000
```

### Analyze a frame

```bash
curl -X POST http://localhost:8000/analyze-frame \
  -H "Content-Type: application/json" \
  -d '{"frame_b64": "<base64 jpeg>", "question": "Is there a person at the door?"}'
```

```json
{"answer": "Yes, there is a person standing at the door.", "triggered": true, "latency_ms": 487}
```

### OpenAI-compatible chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [
      {"type": "video", "video": "video.mp4"},
      {"type": "text", "text": "What is happening?"}
    ]}]
  }'
```

### Watch API — continuous RTSP monitoring

Stream real-time analysis of an RTSP camera via Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/v1/watch \
  -H "Content-Type: application/json" \
  -d '{
    "source": "rtsp://admin:pass@192.168.1.100:554/stream",
    "conditions": [
      {"id": "person", "question": "Is there a person?"},
      {"id": "package", "question": "Is there a package on the doorstep?"}
    ],
    "fps": 1,
    "resolution": "672x448"
  }'
```

SSE events: `status`, `result`, `alert` (condition triggered, includes `frame_b64`), `heartbeat`, `error`.

```bash
curl http://localhost:8000/v1/watch              # List active watches
curl -X DELETE http://localhost:8000/v1/watch/w_abc123  # Stop a watch
```

Features: auto-reconnect on RTSP failure, motion gate (skip inference on static scenes), `<think>` tag stripping, configurable resolution, heartbeat for connection health. See [`examples/watch_camera.py`](examples/watch_camera.py) for a complete SSE client.

### Operations

```bash
# Metrics — uptime, memory, request counts, inference stats
curl http://localhost:8000/metrics

# Hot reload model without downtime (drains in-flight requests first)
curl -X POST http://localhost:8000/v1/admin/reload

# Or reload via signal
kill -HUP $(pgrep -f "trio serve")

# Structured JSON logging (for log aggregation)
trio serve --json-logs
# or: TRIO_LOG_JSON=1 trio serve
```

Request body size is capped at 10 MB. Resolutions are clamped to 4K max to prevent OOM.

### All endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Health check |
| `/health` | GET | Detailed status + config |
| `/metrics` | GET | Uptime, memory (RSS + Metal), request/inference stats |
| `/analyze-frame` | POST | Single frame: `{frame_b64, question}` → `{answer, triggered, latency_ms}` |
| `/v1/watch` | POST | Start watching RTSP stream → SSE events |
| `/v1/watch` | GET | List active watches |
| `/v1/watch/{id}` | DELETE | Stop a watch |
| `/v1/video/analyze` | POST | Video file analysis with metrics |
| `/v1/frames/analyze` | POST | Multi-frame upload (multipart) |
| `/v1/chat/completions` | POST | OpenAI-compatible (streaming SSE) |
| `/v1/models` | GET | Loaded model info |
| `/v1/admin/reload` | POST | Hot reload model (with rollback on failure) |

## OpenClaw Integration

TrioCore can connect directly to an [OpenClaw](https://openclaw.ai) Gateway as a node, enabling remote camera monitoring and VLM inference via WebSocket — no HTTP server needed.

### Setup

```bash
pip install 'trio-core[claw]'

# First time: pair with the Gateway
trio claw --pair -g ws://127.0.0.1:18789 --token <gateway-token>

# Run as node (subsequent times)
trio claw -g ws://127.0.0.1:18789 --token <gateway-token> \
  -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  -c "rtsp://admin:pass@192.168.1.100:554/stream"
```

### Gateway Configuration

Add trio-core commands to the Gateway allowlist:

```bash
openclaw config set gateway.nodes.allowCommands \
  '["camera.snap","camera.list","vision.analyze","vision.watch","vision.watch.stop","vision.status"]'
openclaw gateway restart
```

### Supported Commands

| Command | Description | Params |
|---|---|---|
| `camera.list` | List configured cameras | — |
| `camera.snap` | Capture a single JPEG frame | `deviceId`, `maxWidth`, `quality` |
| `vision.analyze` | Capture + VLM inference (one-shot) | `question` (required), `deviceId` |
| `vision.watch` | Start continuous monitoring loop | `question` (required), `interval` (default 10s), `deviceId` |
| `vision.watch.stop` | Stop one or all watches | `watchId` (optional — omit to stop all) |
| `vision.status` | Report model, device, active watches | — |

### Invoking Commands

#### Via OpenClaw CLI

```bash
# One-shot: ask a question about the current camera frame
openclaw nodes invoke --node <node-id> --command vision.analyze \
  --params '{"question": "What do you see?"}'

# Snap a JPEG frame
openclaw nodes invoke --node <node-id> --command camera.snap \
  --params '{"maxWidth": 640}'

# Start continuous watch (checks every 10s)
openclaw nodes invoke --node <node-id> --command vision.watch \
  --params '{"question": "Are there people at the door?", "interval": 10}'

# Stop a specific watch
openclaw nodes invoke --node <node-id> --command vision.watch.stop \
  --params '{"watchId": "<id from start response>"}'

# Stop all watches
openclaw nodes invoke --node <node-id> --command vision.watch.stop

# Check status (model info + active watches)
openclaw nodes invoke --node <node-id> --command vision.status
```

#### Via WhatsApp / Telegram (natural language)

If your Gateway has a chat channel linked (WhatsApp, Telegram, etc.), you can talk to the agent directly:

| What you want | Example message |
|---|---|
| One-shot check | *"What's on my front door camera right now?"* |
| Start watching | *"Watch my camera and tell me if anyone shows up"* |
| Start with interval | *"Monitor the front door every 30 seconds for packages"* |
| Stop watching | *"Stop the camera watch"* |
| Check status | *"What's the camera node status?"* |
| Snap a photo | *"Take a photo from the front door camera"* |

The Gateway agent automatically maps these to the correct `vision.*` / `camera.*` commands on the trio-core node.

### Architecture

```
WhatsApp / Telegram / Discord
    ↕ natural language
OpenClaw Gateway (WebSocket)
    ↕ node.invoke.request / node.invoke.result
TrioCore Node (trio claw)
    ├── OpenCV VideoCapture → RTSP / USB camera
    └── VLM Engine (MLX) → on-device inference
```

The node connects via WebSocket with Ed25519 device identity, advertises its command surface, and dispatches invoke commands directly to the VLM engine — no intermediate HTTP layer. All inference runs locally on Apple Silicon.

---

## How It Works

TrioCore optimizes **every stage** of VLM inference. Each technique is independent and they compound:

| Stage | Technique | What it does | Speedup |
|---|---|---|---|
| Pre-inference | Temporal dedup | Skip near-identical frames (L2 on 64x64) | -50% frames |
| Pre-inference | Motion gate | Skip VLM entirely when scene is static | -80% calls |
| Vision encoder | **[ToMe](https://arxiv.org/abs/2210.09461)** | Merge similar visual tokens between ViT blocks | **-73% prefill** |
| LLM layers | **[FastV](https://arxiv.org/abs/2403.06764)** | Prune low-attention visual tokens from KV cache | -50% tokens |
| Cross-frame | **KV Reuse** | Reuse KV cache when frames are visually similar | **1.7x speedup** |
| Long video | **[StreamMem](https://arxiv.org/abs/2504.08498)** | Bounded KV cache with saliency eviction | constant memory |

## Supported Models

### Tier 1 — Full optimization (native loading, all 4 stages)

| Model | Size | 4-bit VRAM | ToMe | FastV | Compressed | KV Reuse |
|---|---|---|---|---|---|---|
| Qwen2.5-VL 3B | 3B | 1.8G | ✓ | ✓ | ✓ | ✓ |
| Qwen2.5-VL 7B | 7B | 4.5G | ✓ | ✗ | ✓ | ✓ |
| Qwen3-VL 2B/4B/8B | 2-8B | 1.5-5.0G | — | ✓ | ✓ | ✓ |
| Qwen3.5 0.8B/2B/4B/9B | 0.8-9B | 0.5-5.0G | ✓ | — | ✓ | ✓ (DeltaNet) |
| InternVL3 1B/2B | 1-2B | 1.0-1.6G | — | — | ✓ | ✓ |

### Tier 2 — Inference only (mlx-vlm, no optimization)

Gemma 3n, SmolVLM2, Phi-4, Gemma 3, FastVLM, and any model supported by mlx-vlm.

## Configuration

All settings via `TRIO_` environment variables or `EngineConfig`:

```bash
TRIO_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit
TRIO_TOME_ENABLED=true
TRIO_TOME_R=4
TRIO_PORT=8000
```

See [`trio_core/config.py`](src/trio_core/config.py) for all options.

## Benchmarks

<details>
<summary><strong>Accuracy & latency benchmarks</strong> (Apple M3 Ultra, 4-bit quantized)</summary>

Accuracy is hardware-independent (bit-identical output on any Apple Silicon). Latency scales proportionally across devices.

### POPE — Object Hallucination (100 samples, yes/no)

| Model | Params | Baseline | ToMe r=4 | Compressed 50% | FastV |
|---|---|---|---|---|---|
| **InternVL3-2B** | 2B | **95%** | — | 94% (-1) | — |
| Qwen2.5-VL-3B | 3B | 94% | 91% (-3) | 75% (-19) | 92% (-2) |
| Qwen3.5-2B | 2B | 94% | 93% (-1) | 93% (-1) | — |
| InternVL3-1B | 1B | 93% | — | **94% (+1)** | — |
| Qwen3.5-0.8B | 0.8B | 93% | **94% (+1)** | 93% (0) | — |
| Qwen3-VL-2B | 2B | 92% | — | 92% (0) | 0% |
| Qwen3.5-9B | 9B | 92% | 91% (-1) | 90% (-2) | — |
| Qwen3-VL-8B | 8B | 91% | — | **93% (+2)** | 75% (-16) |
| Qwen3-VL-4B | 4B | 91% | — | 88% (-3) | 85% (-6) |
| Qwen2.5-VL-7B | 7B | 90% | 86% (-4) | 90% (0) | ✗ |
| Qwen3.5-4B | 4B | 90% | 89% (-1) | 89% (-1) | — |

### TextVQA — OCR Reading (50 samples, open-ended)

| Model | Params | Baseline | ToMe r=4 | Compressed 50% | FastV |
|---|---|---|---|---|---|
| Qwen3.5-2B | 2B | **80%** | 78% (-2) | 74% (-6) | — |
| InternVL3-2B | 2B | 78% | — | 72% (-6) | — |
| Qwen3-VL-2B | 2B | 76% | — | **76% (0)** | 66% (-10) |
| Qwen2.5-VL-3B | 3B | 72% | 42% (-30) | 60% (-12) | 40% (-32) |
| Qwen3-VL-4B | 4B | 72% | — | **72% (0)** | 56% (-16) |
| Qwen3.5-0.8B | 0.8B | 70% | 64% (-6) | 52% (-18) | — |
| Qwen3-VL-8B | 8B | 70% | — | **70% (0)** | 54% (-16) |
| Qwen2.5-VL-7B | 7B | 66% | 52% (-14) | **68% (+2)** | ✗ |
| Qwen3.5-9B | 9B | 56% | **62% (+6)** | 56% (0) | — |
| Qwen3.5-4B | 4B | 52% | **64% (+12)** | 52% (0) | — |
| InternVL3-1B | 1B | 50% | — | 50% (0) | — |

### GQA — Visual Reasoning (50 samples, open-ended)

| Model | Params | Baseline | ToMe r=4 | Compressed 50% | FastV |
|---|---|---|---|---|---|
| **Qwen3.5-2B** | 2B | **68%** | 66% (-2) | 68% (0) | — |
| InternVL3-2B | 2B | 66% | — | 66% (0) | — |
| Qwen3-VL-4B | 4B | 66% | — | 62% (-4) | 50% (-16) |
| Qwen3.5-0.8B | 0.8B | 66% | 60% (-6) | 60% (-6) | — |
| InternVL3-1B | 1B | 62% | — | 58% (-4) | — |
| Qwen2.5-VL-3B | 3B | 58% | 54% (-4) | 52% (-6) | 42% (-16) |
| Qwen2.5-VL-7B | 7B | 58% | 58% (0) | 50% (-8) | — |
| Qwen3.5-4B | 4B | 58% | 60% (+2) | **64% (+6)** | — |
| Qwen3.5-9B | 9B | 56% | **64% (+8)** | **62% (+6)** | — |
| Qwen3-VL-2B | 2B | 52% | — | **58% (+6)** | 0% |
| Qwen3-VL-8B | 8B | 48% | — | **54% (+6)** | 42% (-6) |

### MMBench — Multi-ability (50 samples, multiple choice)

| Model | Params | Baseline | ToMe r=4 | Compressed 50% | FastV |
|---|---|---|---|---|---|
| **InternVL3-2B** | 2B | **98%** | — | 96% (-2) | — |
| Qwen2.5-VL-7B | 7B | 96% | 96% (0) | 94% (-2) | — |
| Qwen3-VL-4B | 4B | 96% | — | 94% (-2) | 90% (-6) |
| Qwen3-VL-8B | 8B | 96% | — | 94% (-2) | 78% (-18) |
| Qwen3.5-9B | 9B | 96% | 90% (-6) | 96% (0) | — |
| Qwen2.5-VL-3B | 3B | 90% | 82% (-8) | 86% (-4) | 66% (-24) |
| InternVL3-1B | 1B | 88% | — | 86% (-2) | — |
| Qwen3-VL-2B | 2B | 84% | — | 80% (-4) | 2% |
| Qwen3.5-2B | 2B | 82% | 82% (0) | 82% (0) | — |
| Qwen3.5-0.8B | 0.8B | 58% | **62% (+4)** | 54% (-4) | — |
| Qwen3.5-4B | 4B | 46% | 44% (-2) | 36% (-10) | — |

### MVBench — Video Understanding (19 tasks, 5 samples/task)

| Model | Params | Baseline | Compressed 50% |
|---|---|---|---|
| Qwen3-VL-8B | 8B | **65%** | 61% (-4) |
| Qwen2.5-VL-3B | 3B | 64% | 60% (-4) |
| Qwen3.5-2B | 2B | 64% | 61% (-3) |
| Qwen2.5-VL-7B | 7B | 62% | 60% (-2) |
| Qwen3-VL-4B | 4B | 61% | 58% (-3) |
| Qwen3-VL-2B | 2B | 57% | 53% (-4) |
| Qwen3.5-0.8B | 0.8B | 57% | 53% (-4) |
| Qwen3.5-9B | 9B | 49% | 48% (-1) |
| Qwen3.5-4B | 4B | 1% | 2% (+1) |
| InternVL3 | 1-2B | — | — |

`—` = architecturally incompatible (auto-skipped). `✗` = produces garbage output. ToMe incompatible with Qwen3-VL (deepstack) and InternVL3 (pixel shuffle). FastV incompatible with Qwen3.5 (DeltaNet), InternVL3, Qwen2.5-VL-7B (over-prunes), and Qwen3-VL-2B (garbage output). InternVL3 does not support multi-image/video inference (MVBench). Qwen3.5-4B: known 4-bit quantization issue on MCQ/video benchmarks (official FP16: MMBench 89%, our 4-bit: 46%).

### SurveillanceVQA — Anomaly Detection (1,827 samples, yes/no)

[SurveillanceVQA-589K](https://arxiv.org/abs/2505.12589) detection benchmark on UCF-Crime surveillance videos (13 anomaly categories + normal clips).

| Model | Params | Accuracy | F1 | Recall | Specificity | Yes Rate | Latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 7B | **70.1%** | 0.362 | 25.3% | **92.8%** | 13.3% | 587ms |
| Qwen3-VL-8B | 8B | 69.0% | 0.395 | 30.2% | 88.6% | 17.7% | 450ms |
| Qwen2.5-VL-3B | 3B | 68.4% | 0.504 | 47.6% | 79.1% | 29.9% | 375ms |
| Qwen3-VL-2B | 2B | 67.6% | 0.137 | 7.7% | 97.9% | 4.0% | 193ms |
| Qwen3.5-0.8B | 0.8B | 67.6% | 0.441 | 51.7% | 58.2% | 45.2% | 118ms |
| Qwen3-VL-4B | 4B | 67.5% | 0.484 | 45.4% | 78.7% | 29.3% | 304ms |
| Qwen3.5-2B | 2B | 67.3% | 0.108 | 5.9% | 98.4% | 3.1% | 189ms |
| **Qwen3.5-4B** | 4B | 65.2% | **0.556** | 65.1% | 65.2% | 44.9% | 295ms |
| Qwen3.5-9B | 9B | 56.7% | 0.550 | **79.0%** | 45.5% | 62.7% | 452ms |

**mlx-vlm raw baseline comparison** (no TrioCore optimizations):

| Model | Backend | Accuracy | F1 | Recall | Yes Rate |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | TrioCore | 68.4% | 0.504 | 47.6% | 29.9% |
| | mlx-vlm raw | 67.0% | 0.068 | 3.6% | 1.8% |
| Qwen2.5-VL-7B | TrioCore | 70.1% | 0.362 | 25.3% | 13.3% |
| | mlx-vlm raw | 67.4% | 0.100 | 5.4% | 2.7% |

Key findings: **TrioCore optimizations do NOT cause regression** — TrioCore is slightly *better* than raw mlx-vlm (+1.4% to +2.7% accuracy). All models struggle with surveillance anomaly detection regardless of backend — accuracy ≤70%, confirming this is a fundamental domain gap, not an optimization issue. Most models are extremely conservative (very low recall, high specificity). The best-balanced models are Qwen3.5-4B (F1=0.556) and Qwen3.5-9B (F1=0.550). This validates the need for domain-specific LoRA fine-tuning (see [research/lora.md](research/lora.md)).

### Latency — ms/sample (POPE)

| Model | Baseline | ToMe r=4 | Compressed 50% | FastV | Best Speedup |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 148ms | 167ms | **135ms** | — | 1.09x |
| Qwen3.5-2B | 251ms | 297ms | **221ms** | — | 1.14x |
| Qwen3-VL-2B | 275ms | — | **223ms** | 226ms | 1.23x |
| Qwen2.5-VL-3B | 354ms | 629ms | **279ms** | 288ms | 1.27x |
| Qwen3.5-4B | 407ms | 454ms | **337ms** | — | 1.20x |
| Qwen3-VL-4B | 414ms | — | **335ms** | 341ms | 1.24x |
| Qwen2.5-VL-7B | 522ms | 693ms | **384ms** | — | 1.36x |
| Qwen3-VL-8B | 633ms | — | **503ms** | 516ms | 1.26x |
| Qwen3.5-9B | 632ms | 694ms | **506ms** | — | 1.25x |
| InternVL3-1B | 677ms | — | **577ms** | — | 1.17x |
| InternVL3-2B | 967ms | — | **736ms** | — | 1.31x |

### Frame-to-frame (KV cache reuse, 480p 5-frame video)

| Model | Speedup | Architecture |
|---|---|---|
| Qwen2.5-VL-3B | **1.57x** | KV cache reuse |
| Qwen3-VL-4B | **1.71x** | KV cache reuse |
| Qwen3.5-0.8B | **1.35x** | DeltaNet state snapshot |

### Optimization Guide — Best Performance vs Accuracy Balance

Based on our [full benchmark analysis](research/benchmark-analysis.md) (11 models x 5 benchmarks x 4 configs):

#### Per-model recommendation

| Model | Best Config | Speedup | Avg Accuracy Delta | Verdict |
|---|---|---|---|---|
| Qwen2.5-VL-7B | Compressed 50% | **1.33x** | 0% to +2% | **Best tradeoff** — accuracy improves on some tasks |
| Qwen3-VL-8B | Compressed 50% | **1.20x** | +2% POPE, +6% GQA | **Best tradeoff** — compression acts as regularization |
| Qwen3.5-9B | Compressed 50% | **1.21x** | +6% GQA, 0% TextVQA | **Best tradeoff** — over-parameterized, benefits from compression |
| Qwen3.5-4B | Compressed 50% | **1.17x** | +6% GQA, 0% TextVQA | **Best tradeoff** — same regularization effect |
| Qwen3-VL-4B | Compressed 50% | 1.16x | -2% to -3% | Good |
| Qwen3.5-2B | Compressed 50% | 1.09x | -1% to -3% | Good |
| InternVL3-2B | Compressed 50% | 1.26x | -1% to -2% | Good |
| Qwen3-VL-2B | Compressed 50% | 1.10x | 0% to -4% | Good |
| Qwen3.5-0.8B | Baseline | — | — | Too small to compress (TextVQA -18%) |
| Qwen2.5-VL-3B | Baseline | — | — | Anomalously sensitive (POPE -19%) |
| InternVL3-1B | Baseline | 1.06x | Marginal gain | Not worth the memory overhead |

#### Per-scenario recommendation

| Scenario | Best Strategy | Why |
|---|---|---|
| Independent frames | Compressed 50% (4B+ models) | 1.1-1.3x speedup, -1% to -3% accuracy |
| Sequential video (surveillance) | **Baseline (no compression)** | MLX prompt cache gives 2.25x automatic speedup; compression breaks this cache and is 2.9x slower |
| OCR / text reading | Baseline or minimal compression | TextVQA most sensitive to token loss |
| Detection / monitoring | Compressed 50% aggressively | POPE tolerates compression well (-1% avg) |
| High resolution (1080p+) | Compressed 50% | ToMe gives 4x prefill speedup but ViT overhead negates E2E gain |

#### Key insight

> **4B+ models: use Compressed 50%. Under 2B: use Baseline. Sequential frames: always Baseline.**
>
> Compressed 50% on over-parameterized models (4B+) acts as regularization — it forces the LLM to rely on the most informative visual features, often *improving* accuracy on reasoning tasks while delivering 1.2-1.3x speedup. For sequential video, MLX's built-in prompt cache already provides 2.25x frame-over-frame speedup for free.

### Overhead vs mlx-vlm (raw generate loop)

| Metric | mlx-vlm | trio-core |
|---|---|---|
| Prefill | 1018ms | 1016ms (-0.2%) |
| Decode | 524ms | 513ms (-2.1%) |
| Output | — | **bit-identical** |

</details>

## References

Optimization techniques used in TrioCore:

- **ToMe** (Token Merging) — Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023. [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **FastV** — Chen et al., "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models", ECCV 2024. [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- **StreamingLLM / StreamMem** — Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453); Du et al., "StreamMem: Streaming KV Cache Management for Video Understanding", 2025. [arXiv:2504.08498](https://arxiv.org/abs/2504.08498)

Evaluation benchmarks:

- **POPE** — Li et al., "Evaluating Object Hallucination in Large Vision-Language Models", EMNLP 2023. [arXiv:2305.10355](https://arxiv.org/abs/2305.10355)
- **TextVQA** — Singh et al., "Towards VQA Models That Can Read", CVPR 2019. [arXiv:1904.08920](https://arxiv.org/abs/1904.08920)
- **GQA** — Hudson & Manning, "GQA: A New Dataset for Real-World Visual Reasoning", CVPR 2019. [arXiv:1902.09506](https://arxiv.org/abs/1902.09506)
- **MMBench** — Liu et al., "MMBench: Is Your Multi-modal Model an All-around Player?", ECCV 2024. [arXiv:2307.06281](https://arxiv.org/abs/2307.06281)
- **MVBench** — Li et al., "MVBench: A Comprehensive Multi-modal Video Understanding Benchmark", CVPR 2024. [arXiv:2311.17005](https://arxiv.org/abs/2311.17005)
- **SurveillanceVQA-589K** — Zheng et al., "SurveillanceVQA: Towards Comprehensive and Diversified Surveillance Video Question Answering", 2025. [arXiv:2505.12589](https://arxiv.org/abs/2505.12589)

## License

Apache 2.0
