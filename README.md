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

## How It Works

TrioCore optimizes **every stage** of VLM inference. Each technique is independent and they compound:

| Stage | Technique | What it does | Speedup |
|---|---|---|---|
| Pre-inference | Temporal dedup | Skip near-identical frames (L2 on 64x64) | -50% frames |
| Pre-inference | Motion gate | Skip VLM entirely when scene is static | -80% calls |
| Vision encoder | **ToMe** | Merge similar visual tokens between ViT blocks | **-73% prefill** |
| LLM layers | **FastV** | Prune low-attention visual tokens from KV cache | -50% tokens |
| Cross-frame | **KV Reuse** | Reuse KV cache when frames are visually similar | **1.7x speedup** |
| Long video | **StreamMem** | Bounded KV cache with saliency eviction | constant memory |

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

### MVBench — Video Understanding (12 tasks, 5 samples/task)

| Model | Params | Baseline | Compressed 50% |
|---|---|---|---|
| Qwen3-VL-8B | 8B | **69%** | 57% (-12) |
| Qwen3.5-2B | 2B | 65% | 57% (-8) |
| Qwen2.5-VL-7B | 7B | 63% | 61% (-2) |
| Qwen3-VL-2B | 2B | 63% | 54% (-9) |
| Qwen3-VL-4B | 4B | 63% | 54% (-9) |
| Qwen2.5-VL-3B | 3B | 61% | 59% (-2) |
| Qwen3.5-0.8B | 0.8B | 50% | 46% (-4) |
| Qwen3.5-9B | 9B | 37% | 37% (0) |
| Qwen3.5-4B | 4B | 2% | 2% (0) |
| InternVL3 | 1-2B | — | — |

`—` = architecturally incompatible (auto-skipped). `✗` = produces garbage output. ToMe incompatible with Qwen3-VL (deepstack) and InternVL3 (pixel shuffle). FastV incompatible with Qwen3.5 (DeltaNet), InternVL3, Qwen2.5-VL-7B (over-prunes), and Qwen3-VL-2B (garbage output). InternVL3 does not support multi-image/video inference (MVBench). Qwen3.5-4B: known 4-bit quantization issue on MCQ/video benchmarks (official FP16: MMBench 89%, our 4-bit: 46%).

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

### Overhead vs mlx-vlm (raw generate loop)

| Metric | mlx-vlm | trio-core |
|---|---|---|
| Prefill | 1018ms | 1016ms (-0.2%) |
| Decode | 524ms | 513ms (-2.1%) |
| Output | — | **bit-identical** |

</details>

## License

Apache 2.0
