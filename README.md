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
  <a href="#install">Install</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#api-server">API Server</a> |
  <a href="#how-it-works">How It Works</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#supported-models">Models</a>
</p>

---

## Install

```bash
pip install 'trio-core[mlx]'       # Apple Silicon (M1-M4)
pip install 'trio-core[transformers]'  # NVIDIA / CPU
```

## Quick Start

### CLI

```bash
# Check your hardware
trio device

# Analyze a video
trio analyze video.mp4 -q "What is happening?"

# Start the API server
trio serve
```

### Python

```python
from trio_core import TrioCore

engine = TrioCore()
engine.load()

result = engine.analyze_video("clip.mp4", "What is happening?")
print(result.text)  # "A person is walking through the parking lot..."
print(f"{result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.0f} tok/s")
```

### With optimizations

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

### Analyze a video

```bash
curl -X POST http://localhost:8000/v1/video/analyze \
  -H "Content-Type: application/json" \
  -d '{"video": "video.mp4", "prompt": "What is happening?"}'
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

### All endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Health check |
| `/health` | GET | Detailed status + config |
| `/analyze-frame` | POST | Single frame: `{frame_b64, question}` → `{answer, triggered, latency_ms}` |
| `/v1/video/analyze` | POST | Video file analysis with metrics |
| `/v1/frames/analyze` | POST | Multi-frame upload (multipart) |
| `/v1/chat/completions` | POST | OpenAI-compatible (streaming SSE) |
| `/v1/models` | GET | Loaded model info |

## How It Works

TrioCore optimizes **every stage** of VLM inference. Each technique is independent and they compound:

```
Video → [Dedup] → [Motion Gate] → [ViT + ToMe] → [LLM + FastV] → [KV Reuse] → Answer
         -50%       -80% calls     -68% tokens    -50% tokens     1.7x reuse
        frames      when static    in encoder      in LLM         across frames
```

| Stage | Technique | What it does | Speedup |
|---|---|---|---|
| Pre-inference | Temporal dedup | Skip near-identical frames (L2 on 64x64) | -50% frames |
| Pre-inference | Motion gate | Skip VLM entirely when scene is static | -80% calls |
| Vision encoder | **ToMe** | Merge similar visual tokens between ViT blocks | **-73% prefill** |
| LLM layers | **FastV** | Prune low-attention visual tokens from KV cache | -50% tokens |
| Cross-frame | **KV Reuse** | Reuse KV cache when frames are visually similar | **1.7x speedup** |
| Long video | **StreamMem** | Bounded KV cache with saliency eviction | constant memory |

## Benchmarks

Apple M3 Pro, 4-bit quantized.

### Prefill speed (1080p single frame)

| Model | Baseline | + ToMe r=4 | Speedup |
|---|---|---|---|
| Qwen2.5-VL-3B | 1,808ms (748 tokens) | 490ms (242 tokens) | **3.7x** |

### Quality (POPE, 100 images)

| Model | Baseline | + ToMe r=4 |
|---|---|---|
| Qwen2.5-VL-3B | 92% | 81% |
| Qwen3-VL-4B | 91% | **91% (zero loss)** |

### Frame-to-frame (480p, 5-frame video)

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

## Supported Models

### Tier 1 — Full optimization (native loading, all 4 stages)

| Model | Size | 4-bit | ToMe | FastV | KV Reuse | StreamMem |
|---|---|---|---|---|---|---|
| Qwen2.5-VL 3B/7B | 3-7B | 1.8-4.5G | ✓ | ✓ | ✓ | ✓ |
| Qwen3-VL 2B/4B/8B | 2-8B | 1.5-5.0G | ✓ | ✓ | ✓ | ✓ |
| Qwen3.5 0.8B/2B/4B/9B | 0.8-9B | 0.5-5.0G | ✓ | ✓ | ✓ (DeltaNet) | ✓ |
| InternVL3 1B/2B | 1-2B | 1.0-1.6G | — | ✓ | ✓ | ✓ |
| nanoLLaVA-1.5 | 1B | 1.0G | ✓ | ✓ | ✓ | ✓ |

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

## License

Apache 2.0
