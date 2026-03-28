<p align="center">
  <h1 align="center">TrioCore</h1>
  <p align="center">
    <strong>Real-time Vision Intelligence Engine for Apple Silicon</strong>
  </p>
  <p align="center">
    YOLO object detection + VLM scene understanding. One pip install, zero Docker.
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
  <a href="#install">Install</a> |
  <a href="#api-reference">API</a> |
  <a href="#cli">CLI</a> |
  <a href="#python-sdk">SDK</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## What is TrioCore?

Point it at any image, video, or camera and it will detect objects, count people, and describe scenes — all running locally on your Mac, no cloud APIs needed.

**Core capabilities:**
- **Detect** — Find and count objects (people, cars, etc.) in images
- **Describe** — Get natural language descriptions of what's happening in a scene
- **Crop-Describe** — Detect objects, then describe each one individually
- **REST API** — Built-in web server on port 8100 with interactive docs
- **CLI** — Simple commands: `trio serve`, `trio analyze`, `trio webcam`

<details>
<summary><strong>New to computer vision? Key terms explained</strong></summary>

| Term | What it means |
|---|---|
| **YOLO** | "You Only Look Once" — a fast object detection model that finds and labels objects in images |
| **VLM** | Vision Language Model — an AI model that can look at an image and describe it in natural language |
| **MLX** | Apple's machine learning framework, optimized for M1/M2/M3/M4 chips |
| **ONNX** | A standard format for ML models that runs on any hardware |
| **ToMe** | Token Merging — a technique that makes VLM inference faster by reducing redundant data |
| **KV cache** | A memory optimization that speeds up processing of sequential video frames |

</details>

---

## Quick Start

```bash
# 1. Install (Apple Silicon Mac recommended)
pip install 'trio-core[mlx]'

# 2. Check your setup
trio doctor

# 3. Start the server
trio serve
```

> **First run note:** The first time you run `trio serve` or `trio analyze`, the model
> will be downloaded automatically (~2 GB for the default 3B model). This takes 5-20
> minutes depending on your connection. Subsequent runs start instantly.

Once the server is running, open **http://localhost:8100/docs** in your browser to
explore the API interactively, or try it from the terminal:

```bash
# In another terminal — grab any image and detect objects in it
# macOS:
curl -X POST http://localhost:8100/api/inference/detect \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "'$(base64 -i your-photo.jpg)'"}'

# Linux:
curl -X POST http://localhost:8100/api/inference/detect \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "'$(base64 -w0 your-photo.jpg)'"}'
```

```json
{
  "people_count": 3,
  "vehicle_count": 1,
  "by_class": {"person": 3, "car": 1},
  "crops_b64": [{"class": "person", "bbox": [100, 50, 200, 300], "confidence": 0.92}],
  "elapsed_ms": 45
}
```

Or analyze an image directly from the CLI (no server needed):

```bash
trio analyze your-photo.jpg -q "How many people are in this image?"
```

See more in [`examples/`](examples/) — [`quickstart.py`](examples/quickstart.py) (5 lines)
and [`api_client.py`](examples/api_client.py) (full API usage).

---

## Install

Requires **Python 3.12+**.

```bash
# Apple Silicon Mac (M1/M2/M3/M4) — recommended, uses Apple's MLX framework
pip install 'trio-core[mlx]'

# Apple Silicon + webcam monitoring
pip install 'trio-core[mlx,webcam]'

# NVIDIA GPU or CPU-only (uses PyTorch/Transformers instead of MLX)
pip install 'trio-core[transformers]'

# For IP/RTSP camera support (macOS)
brew install ffmpeg
```

**Which install do I pick?** If you have a Mac with Apple Silicon (2020 or later), use `[mlx]`. If you have an NVIDIA GPU or are on Linux, use `[transformers]`. Not sure? Run `trio device` after install to see what hardware was detected.

---

## API Reference

> **Tip:** Once the server is running, visit **http://localhost:8100/docs** for interactive
> API documentation where you can try every endpoint from your browser.

Start the server:

```bash
trio serve                          # default: 0.0.0.0:8100
trio serve --port 9000              # custom port
TRIO_API_KEY=secret trio serve      # enable Bearer token auth
```

### `POST /api/inference/detect`

Run YOLO object detection. Returns counts and bounding boxes.

```bash
curl -X POST http://localhost:8100/api/inference/detect \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64 jpeg>", "pad_ratio": 0.15}'
```

**Response:**
```json
{
  "people_count": 2,
  "vehicle_count": 1,
  "by_class": {"person": 2, "car": 1},
  "crops_b64": [
    {"class": "person", "bbox": [100, 50, 200, 300], "confidence": 0.92},
    {"class": "car", "bbox": [400, 200, 600, 350], "confidence": 0.87}
  ],
  "elapsed_ms": 42
}
```

### `POST /api/inference/describe`

Run VLM on an image. Returns natural language description.

```bash
curl -X POST http://localhost:8100/api/inference/describe \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64 jpeg>", "prompt": "Describe what you see."}'
```

**Response:**
```json
{
  "description": "A woman in a red jacket is walking a golden retriever along a tree-lined sidewalk.",
  "elapsed_ms": 380
}
```

### `POST /api/inference/crop-describe`

Combined pipeline: YOLO detects objects, crops them, then VLM describes each entity individually before generating a full scene description.

```bash
curl -X POST http://localhost:8100/api/inference/crop-describe \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "<base64 jpeg>",
    "crops": [
      {"class": "person", "bbox": [100, 50, 200, 300], "confidence": 0.92}
    ],
    "max_crops": 3
  }'
```

**Response:**
```json
{
  "description": "1 person: male 30s, blue polo, carrying laptop bag",
  "entities": {"persons": [...], "vehicles": [...]},
  "crop_descriptions": ["person: male 30s, blue polo, carrying laptop bag"],
  "elapsed_ms": 520
}
```

### `GET /api/inference/status`

Check which models are loaded.

### `GET /health`

Health check with uptime.

---

## CLI

```bash
trio doctor                             # Check setup — run this first!
trio device                             # Show your hardware + recommended model
trio serve                              # Start inference API server (port 8100)
trio analyze photo.jpg -q "What's here?" # Analyze an image (no server needed)
trio analyze video.mp4 -q "Describe"    # Video analysis
trio webcam -w "a person is waving"     # Live webcam monitor with alerts
trio cam --host 192.168.1.100 -p pass   # IP camera monitor
trio bench video.mp4 -n 5              # Benchmark inference speed
```

### `trio analyze`

```bash
trio analyze photo.jpg -q "How many people are in this image?"
trio analyze video.mp4 -q "Describe the scene" --json    # JSON output with metrics
trio analyze photo.jpg -m mlx-community/Qwen2.5-VL-7B-Instruct-4bit  # specific model
```

### `trio webcam`

Live camera monitor with VLM-based alerting. Green = clear, red = alert with audio.

```bash
trio webcam -w "someone at the door"         # Built-in webcam
trio webcam -s 1 -w "package on doorstep"    # iPhone Continuity Camera
trio webcam --count                          # Count objects (cumulative)
```

---

## Python SDK

```python
from trio_core import TrioCore, EngineConfig

# Load with defaults (auto-selects best model for your hardware)
engine = TrioCore()
engine.load()

# Analyze an image or video
result = engine.analyze_video("photo.jpg", "What do you see?")
print(result.text)
print(f"{result.metrics.latency_ms:.0f}ms | {result.metrics.tokens_per_sec:.0f} tok/s")
```

### Configuration

```python
config = EngineConfig(
    model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    tome_enabled=True,       # Token Merging — 73% fewer visual tokens
    tome_r=4,
)
engine = TrioCore(config)
```

Or via environment variables:

```bash
TRIO_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit
TRIO_TOME_ENABLED=true
TRIO_TOME_R=4
```

---

## Supported Models

### Tier 1 — Full optimization (native loading + visual token compression + KV reuse)

| Model | Params | 4-bit VRAM | ToMe | Compressed | KV Reuse |
|---|---|---|---|---|---|
| Qwen2.5-VL | 3B, 7B | 1.8-4.5G | yes | yes | yes |
| Qwen3-VL | 2B, 4B, 8B | 1.5-5.0G | -- | yes | yes |
| Qwen3.5 | 0.8-9B | 0.5-5.0G | yes | yes | yes |
| InternVL3 | 1B, 2B | 1.0-1.6G | -- | yes | yes |

### Tier 2 — Inference only (via mlx-vlm)

Gemma 3n, SmolVLM2, Phi-4, FastVLM, and any model supported by mlx-vlm.

---

## Benchmarks

All benchmarks on Apple M3 Ultra, 4-bit quantized models. Accuracy is hardware-independent.

### Inference Latency (POPE benchmark, ms/sample)

| Model | Params | Baseline | Compressed 50% | Speedup |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | 148ms | **135ms** | 1.09x |
| Qwen3.5-2B | 2B | 251ms | **221ms** | 1.14x |
| Qwen3-VL-2B | 2B | 275ms | **223ms** | 1.23x |
| Qwen2.5-VL-3B | 3B | 354ms | **279ms** | 1.27x |
| Qwen2.5-VL-7B | 7B | 522ms | **384ms** | 1.36x |
| Qwen3-VL-8B | 8B | 633ms | **503ms** | 1.26x |

### Frame-to-Frame KV Cache Reuse

| Model | Speedup | Method |
|---|---|---|
| Qwen3-VL-4B | **1.71x** | KV cache reuse |
| Qwen2.5-VL-3B | **1.57x** | KV cache reuse |
| Qwen3.5-0.8B | **1.35x** | DeltaNet state snapshot |

### Overhead vs raw mlx-vlm

| Metric | mlx-vlm | trio-core | Delta |
|---|---|---|---|
| Prefill | 1018ms | 1016ms | -0.2% |
| Decode | 524ms | 513ms | -2.1% |
| Output | -- | **bit-identical** | -- |

<details>
<summary><strong>Full accuracy benchmarks (11 models x 6 benchmarks)</strong></summary>

### POPE — Object Hallucination (100 samples)

| Model | Baseline | Compressed 50% |
|---|---|---|
| InternVL3-2B | **95%** | 94% |
| Qwen2.5-VL-3B | 94% | 75% |
| Qwen3.5-2B | 94% | 93% |
| Qwen3-VL-8B | 91% | **93%** |

### TextVQA — OCR Reading (50 samples)

| Model | Baseline | Compressed 50% |
|---|---|---|
| Qwen3.5-2B | **80%** | 74% |
| InternVL3-2B | 78% | 72% |
| Qwen3-VL-2B | 76% | **76%** |

### GQA — Visual Reasoning (50 samples)

| Model | Baseline | Compressed 50% |
|---|---|---|
| Qwen3.5-2B | **68%** | **68%** |
| InternVL3-2B | 66% | 66% |
| Qwen3.5-4B | 58% | **64%** |

### MMBench — Multi-ability (50 samples)

| Model | Baseline | Compressed 50% |
|---|---|---|
| InternVL3-2B | **98%** | 96% |
| Qwen2.5-VL-7B | 96% | 94% |
| Qwen3.5-9B | 96% | 96% |

### SurveillanceVQA — Anomaly Detection (1,827 samples)

| Model | Accuracy | F1 | Recall |
|---|---|---|---|
| Qwen2.5-VL-7B | **70.1%** | 0.362 | 25.3% |
| Qwen3-VL-8B | 69.0% | 0.395 | 30.2% |
| Qwen3.5-4B | 65.2% | **0.556** | 65.1% |

</details>

---

## Architecture

```
                           TrioCore
                              |
              +---------------+---------------+
              |                               |
         YOLO Pipeline                   VLM Pipeline
              |                               |
    YOLOv10n ONNX model              Qwen/InternVL (MLX)
    tiled 2x2 detection              native model loading
    ByteTrack tracking               ToMe token compression
              |                       KV cache reuse
              |                               |
              +---------------+---------------+
                              |
                    FastAPI Server (:8100)
                              |
              +-------+-------+-------+
              |       |       |       |
          /detect  /describe  /crop   /status
                              -describe
```

### Key design decisions

- **No ultralytics** — YOLOv10 loaded via ONNX Runtime (MIT license)
- **Native VLM loading** — Vendored model code (~3600 lines), bit-identical with mlx-vlm, zero overhead
- **Visual token compression** — ToMe merges similar visual tokens in the ViT, reducing prefill by up to 73%
- **KV cache reuse** — For sequential frames, reuse KV cache from previous frame (1.7x speedup)
- **Lazy loading** — Models loaded on first request, not at server start

---

## Configuration

All settings via environment variables or `EngineConfig`:

| Variable | Default | Description |
|---|---|---|
| `TRIO_MODEL` | Auto-detected | HuggingFace model ID |
| `TRIO_TOME_ENABLED` | `false` | Enable Token Merging |
| `TRIO_TOME_R` | `4` | Tokens merged per ViT block |
| `TRIO_COMPRESS_ENABLED` | `false` | Enable visual token compression |
| `TRIO_COMPRESS_RATIO` | `0.5` | Compression ratio |
| `TRIO_API_KEY` | (none) | Bearer token for API auth |
| `TRIO_YOLO_MODEL` | (auto-downloaded) | Path to YOLO ONNX model |

See [`src/trio_core/config.py`](src/trio_core/config.py) for all options.

---

## OpenClaw Integration

TrioCore can connect to an [OpenClaw](https://openclaw.ai) Gateway as a node for remote camera monitoring via WebSocket.

```bash
pip install 'trio-core[claw]'
trio claw --pair -g ws://gateway:18789 --token <secret>
trio claw -g ws://gateway:18789 -c "rtsp://admin:pass@camera/stream"
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `trio serve` hangs on first run | It's downloading the VLM (~2 GB) and YOLO model (~9 MB) automatically. Wait for it to finish. Check progress with `ls -la ~/.cache/huggingface/` |
| `ModuleNotFoundError: mlx` | You installed without the `[mlx]` extra. Run `pip install 'trio-core[mlx]'` |
| Server starts but curl returns errors | Make sure you're using port **8100** (not 8000). Check with `curl http://localhost:8100/health` |
| `trio analyze` says "no model found" | Run `trio doctor` to check your setup and see which models are available |
| Out of memory on large images | Try a smaller model: `trio serve` defaults to a 3B model (~2 GB RAM). The 7B model needs ~5 GB |
| Webcam not detected | On macOS, grant Terminal camera access in System Settings > Privacy > Camera |

Run `trio doctor` to diagnose most issues — it checks Python version, dependencies, hardware, and available models.

---

## References

- **ToMe** — Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023. [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **StreamMem** — Du et al., "Streaming KV Cache Management for Video Understanding", 2025. [arXiv:2504.08498](https://arxiv.org/abs/2504.08498)
- **SurveillanceVQA-589K** — Zheng et al., 2025. [arXiv:2505.12589](https://arxiv.org/abs/2505.12589)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
