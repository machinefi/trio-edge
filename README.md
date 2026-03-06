<p align="center">
  <h1 align="center">TrioCore</h1>
  <p align="center">
    <strong>Portable video inference engine for Vision Language Models</strong>
  </p>
  <p align="center">
    Runs on Apple Silicon, NVIDIA GPU, and CPU — 4,800 lines, zero Docker.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#examples">Examples</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#api">API</a> |
  <a href="#model-profiles">Model Profiles</a>
</p>

---

## Why TrioCore?

Every VLM engine treats vision as "one image per request." TrioCore treats **video as a first-class input** — with temporal deduplication, motion gating, streaming capture, and visual token compression.

**The thesis:** The biggest win in video VLM isn't faster inference — it's **fewer visual tokens and fewer VLM calls**. TrioCore's optimization stack compounds at every layer:

```
Optimization Stack (all layers compound):
  Frame-level:  Motion Gate + Temporal Dedup ──→ 50-90% fewer VLM calls
  Token-level:  ToMe (inside ViT) ─────────────→ up to 68% fewer visual tokens
  Compute:      Quantization (INT4) ───────────→ cheaper per-token math
```

| | mlx-vlm | vllm-mlx | Roboflow Inference | **TrioCore** |
|---|---|---|---|---|
| Lines of code | 66K | 37K | 370K | **4,800** |
| Video-first | Partial | No | No | **Yes** |
| Stream capture | No | No | No | **Yes** |
| Temporal dedup | No | No | Coarse | **Yes** |
| Motion gating | No | No | No | **Yes** |
| Visual token compression | No | No | No | **Yes (ToMe)** |
| Apple Silicon | Yes | Yes | No | **Yes** |
| Docker required | No | No | Yes | **No** |

## Quick Start

```bash
# Install
pip install 'trio-core[mlx]'           # Apple Silicon (M1-M4)
pip install 'trio-core[transformers]'   # NVIDIA GPU / CPU
pip install 'trio-core[all]'            # Both backends

# Check hardware
trio-core device

# Analyze a video
trio-core analyze video.mp4 --prompt "Is anyone wearing a hard hat?"

# Start API server
trio-core serve
```

### Python API

```python
from trio_core import TrioCore

engine = TrioCore()
engine.load()  # auto-detects: M3 -> MLX, RTX 4090 -> Transformers

# Analyze a video file
result = engine.analyze_video("clip.mp4", "What is happening?")
print(result.text)
print(f"Latency: {result.metrics.latency_ms:.0f}ms")

# Analyze a single frame
result = engine.analyze_frame(frame, "Is the door open?")

# Force a specific backend
engine = TrioCore(backend="transformers")
```

## Examples

See [`examples/`](examples/) for complete scripts:

| Example | Description |
|---|---|
| [`webcam_to_text.py`](examples/webcam_to_text.py) | Live laptop camera -> continuous VLM analysis |
| [`video_analyze.py`](examples/video_analyze.py) | Analyze a video file with detailed metrics |
| [`stream_monitor.py`](examples/stream_monitor.py) | Monitor RTSP/YouTube stream with motion gating |
| [`run_benchmark.py`](examples/run_benchmark.py) | POPE/TextVQA benchmarks with A/B comparison |
| [`run_eval.py`](examples/run_eval.py) | Synthetic perf eval (prefill, decode, memory) |

```python
# Webcam -> Text (3 lines of code)
from trio_core import TrioCore, StreamCapture

engine = TrioCore()
engine.load()

with StreamCapture(0, vid_stride=30) as cam:  # webcam, every 30th frame
    for frame in cam:
        result = engine.analyze_frame(frame, "Describe what you see.")
        print(result.text)
```

## Architecture

```
Live Stream / File / URL
        |
   StreamCapture          daemon thread, grab/retrieve skip, dual-mode buffer
        |
   Frame Extraction       OpenCV, smart resize (merge_factor aligned)
        |
   Temporal Dedup         normalized L2 on 64x64 downscale, threshold=0.95
        |
   Motion Gate            frame differencing + EMA background
        |
   Model Profile          auto-select params for Qwen2.5/3/3.5-VL
        |
   Backend Auto-Select    detect_device() -> MLXBackend / TransformersBackend
        |
   ToMe (optional)        bipartite soft matching inside ViT blocks
        |
   VLM Inference          backend.generate(), thread-locked
        |
   Callbacks              on_frame_captured -> on_dedup_done -> on_vlm_end
        |
   VideoResult            text + InferenceMetrics (3-phase timing)
```

### Hardware Auto-Detection

| Hardware | Detection | Backend | Model Registry |
|---|---|---|---|
| Apple Silicon (M1-M4) | `arm64` + `sysctl` | MLX (mlx-vlm) | `mlx-community/` |
| NVIDIA GPU | `nvidia-smi` / `torch.cuda` | Transformers (PyTorch) | `Qwen/` |
| CPU-only | Fallback | Transformers (PyTorch) | `Qwen/` |

Memory-based model recommendation: 32GB+ -> 7B, <32GB -> 3B.

## Key Technologies

### Visual Token Compression (ToMe)

Training-free token merging inside the vision encoder, based on [Token Merging (ICLR 2023)](https://arxiv.org/abs/2210.09461). Bipartite soft matching merges similar visual tokens between ViT blocks, gradually reducing the sequence length. No training required — works with any Qwen2.5-VL or Qwen3-VL checkpoint.

**Key technical contributions:**

1. **In-encoder merging for VLMs** — ToMe was designed for ViT classifiers. We adapted it for VLM vision encoders (Qwen2.5-VL, Qwen3-VL), handling PatchMerger, windowed attention, RoPE, and the vision-language token handoff.

2. **Windowed-attention-aware merging** — Qwen2.5-VL uses windowed attention in early ViT layers. We merge only within each attention window, respecting the model's locality structure.

3. **Compressed position encoding** — After merging reduces token count, we recompute `grid_thw` and RoPE position IDs so the language model sees geometrically consistent positions.

4. **Dual-model support** — Auto-detects Qwen2.5-VL vs Qwen3-VL (which has deepstack features, different config attributes, and a different `merge_input_ids_with_image_features` signature) and adapts the entire pipeline accordingly.

```python
from trio_core import TrioCore, EngineConfig

config = EngineConfig(tome_enabled=True, tome_r=4)
engine = TrioCore(config)
engine.load()
result = engine.analyze_video(frames, "What do you see?")
```

### StreamCapture — Continuous Frame Capture

Daemon thread with dual-mode buffer, inspired by [ultralytics LoadStreams](https://github.com/ultralytics/ultralytics).

- **Latest-frame mode** (default): Overwrites buffer. For real-time monitoring.
- **Queue mode**: Buffers up to 30 frames. For digest/summary jobs.
- **grab/retrieve split**: Zero-cost frame skipping via `vid_stride`.
- **Auto-reconnect**: 5 retries with 1s delay. YouTube via yt-dlp.

```python
with StreamCapture("rtsp://camera/stream", vid_stride=5) as cap:
    for frame in cap:  # yields (C, H, W) float32
        result = engine.analyze_frame(frame, "Is the door open?")
```

### Temporal Deduplication

Consecutive frames are often near-identical. Sending all to VLM wastes compute.

Compare via normalized L2 on 64x64 downscaled grayscale. Skip frames with similarity > 0.95. Typical result: **30-70% frame reduction** with no information loss.

### Motion Gate

For monitoring: skip VLM entirely when the scene is static.

Frame differencing against EMA background model. When a camera watches a mostly-static scene, eliminates **80%+ of VLM calls**.

## Benchmarks

All benchmarks on Apple M3 Pro, 4-bit quantized models.

### POPE — Object Hallucination (100 COCO images)

| Model | Config | Accuracy | F1 | Avg Latency | Visual Tokens |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | Baseline | 92.0% | 0.913 | 952ms | 363 |
| Qwen2.5-VL-3B | ToMe r=4 | 81.0% | 0.800 | 752ms (-21%) | 131 (-64%) |
| **Qwen3-VL-4B** | **Baseline** | **91.0%** | **0.901** | **795ms** | **278** |
| **Qwen3-VL-4B** | **ToMe r=4** | **91.0%** | **0.901** | **777ms (-2%)** | **258 (-7%)** |

Qwen3-VL shows **zero quality loss** with ToMe — its architecture (no windowed attention, deepstack features) is naturally more robust to token merging.

### Synthetic Eval — Prefill Performance at 1080p

High-resolution video is where ToMe shines most. More visual tokens = more to merge.

| Model | Config | Prefill | Visual Tokens | Peak Memory |
|---|---|---|---|---|
| Qwen2.5-VL-3B | Baseline | 1,808ms | 748 | 4.02 GB |
| Qwen2.5-VL-3B | ToMe r=4 | 490ms (**-73%**) | 242 (**-68%**) | 3.85 GB (-4%) |
| Qwen3-VL-4B | Baseline | 835ms | 323 | 4.07 GB |
| Qwen3-VL-4B | ToMe r=4 | 573ms (**-31%**) | 303 (**-6%**) | 3.71 GB (-9%) |

### Eval Framework

Standard VLM benchmarks for measuring quality vs speed tradeoffs:

```bash
# POPE benchmark (object hallucination, yes/no questions)
python examples/run_benchmark.py --bench pope --samples 100

# Compare baseline vs ToMe
python examples/run_benchmark.py --bench pope --samples 100 --tome 4
python examples/run_benchmark.py --compare baseline.json tome.json

# Synthetic performance eval at different resolutions
python examples/run_eval.py --resolution 1080p --tome 4 --runs 3
```

Supported benchmarks:
- **POPE** — Object hallucination (yes/no), accuracy + F1
- **TextVQA** — OCR-based visual QA
- **Custom** — Your own image + question + answer JSON files

See [`research/`](research/) for detailed analysis, implementation notes, and raw eval results.

## Model Profiles

TrioCore is optimized for the Qwen VL family. Each model has different architecture parameters — using the wrong ones wastes context or produces misaligned tensors.

```python
from trio_core import get_profile

profile = get_profile("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
print(profile.merge_factor)       # 28
print(profile.max_visual_tokens)  # 24576
```

| | Qwen3.5-0.8B | Qwen2.5-VL-3B | Qwen3-VL-4B | Qwen2.5-VL-7B |
|---|---|---|---|---|
| **Architecture** | Gated DeltaNet + Attn | Full GQA | Full GQA + Deepstack | Full GQA |
| **Context** | 262K | 128K | 128K | 128K |
| **Patch size** | 16 | 14 | 14 | 14 |
| **merge_factor** | 32 | 28 | 28 | 28 |
| **Max visual tokens** | 8,192 | 24,576 | 24,576 | 24,576 |
| **DeltaNet layers** | 18/24 | 0 | 0 | 0 |
| **KV heads** | 2 | 2 | 4 | 4 |
| **4-bit size** | ~0.5 GB | ~1.8 GB | ~2.5 GB | ~4.5 GB |

The engine automatically selects the correct `merge_factor` for resize alignment and computes optimal `(frames, height, width)` to fit within each model's visual token budget:

```python
# Visual tokens = (frames / temporal_patch) x (H / merge_factor) x (W / merge_factor)
profile.compute_visual_tokens(8, 224, 224)  # 256 tokens

# Auto-optimize dimensions for token budget
frames, h, w = profile.compute_optimal_params(
    duration_sec=30.0, native_height=1080, native_width=1920
)
```

### Callback System

10 lifecycle events. Errors caught and logged — never crash the pipeline.

```python
engine.add_callback("on_vlm_end", lambda e: send_webhook(e.last_result))
engine.add_callback("on_trigger", lambda e: notify_slack(e.last_result.text))
```

| Event | When |
|---|---|
| `on_engine_load` | Model loaded |
| `on_frame_captured` | Raw frames extracted |
| `on_dedup_done` | Temporal dedup complete |
| `on_motion_check` | Motion gate evaluated |
| `on_vlm_start` / `on_vlm_end` | VLM inference lifecycle |
| `on_trigger` | Watch condition met |
| `on_stream_start` / `on_stream_frame` / `on_stream_stop` | Stream lifecycle |

### Three-Phase Pipeline Timing

Every inference call profiled in three phases:

| Phase | What |
|---|---|
| `preprocess_ms` | Frame extraction, dedup, motion gate |
| `inference_ms` | VLM forward pass (thread-locked) |
| `postprocess_ms` | Metrics assembly |

```python
result = engine.analyze_video("clip.mp4", "Describe this.")
m = result.metrics
print(f"Pre: {m.preprocess_ms:.0f}ms  Inf: {m.inference_ms:.0f}ms  Post: {m.postprocess_ms:.0f}ms")
```

## API

```bash
trio-core serve --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health + model status + profile info |
| `/v1/models` | GET | Loaded model info |
| `/v1/video/analyze` | POST | Video analysis with metrics |
| `/v1/chat/completions` | POST | OpenAI-compatible (SSE streaming) |

### Video Analyze

```bash
curl -X POST http://localhost:8000/v1/video/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4", "prompt": "Is anyone wearing a hard hat?"}'
```

### OpenAI-Compatible Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-3b",
    "messages": [{"role": "user", "content": [
      {"type": "video", "video": "video.mp4"},
      {"type": "text", "text": "What is happening?"}
    ]}]
  }'
```

## Configuration

All settings via environment variables with `TRIO_` prefix:

```bash
TRIO_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit
TRIO_VIDEO_FPS=2.0
TRIO_VIDEO_MAX_FRAMES=128
TRIO_DEDUP_ENABLED=true
TRIO_DEDUP_THRESHOLD=0.95
TRIO_MOTION_ENABLED=false
TRIO_MOTION_THRESHOLD=0.03
TRIO_MAX_TOKENS=512
TRIO_TEMPERATURE=0.0
TRIO_TOME_ENABLED=false
TRIO_TOME_R=4
TRIO_TOME_METRIC=hidden
TRIO_TOME_MIN_KEEP_RATIO=0.3
TRIO_HOST=0.0.0.0
TRIO_PORT=8000
```

Or via Python:

```python
from trio_core import TrioCore, EngineConfig

config = EngineConfig(
    model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    video_fps=1.0,
    dedup_threshold=0.98,
    motion_enabled=True,
    tome_enabled=True,
    tome_r=4,
    max_tokens=1024,
)
engine = TrioCore(config)
```

## Project Structure

```
trio-core/                        ~4,800 lines production code
├── pyproject.toml
├── examples/
│   ├── webcam_to_text.py           Laptop camera -> continuous VLM
│   ├── video_analyze.py            File analysis with metrics
│   ├── stream_monitor.py           Live stream monitoring
│   ├── run_benchmark.py            POPE/TextVQA benchmark runner
│   └── run_eval.py                 Synthetic perf eval suite
├── src/trio_core/
│   ├── __init__.py            33   Public API exports
│   ├── backends.py           453   BaseBackend + MLXBackend + TransformersBackend
│   ├── callbacks.py           80   Hook system (10 events)
│   ├── cli.py                172   serve / analyze / bench / device
│   ├── config.py              58   Pydantic settings (TRIO_ env prefix)
│   ├── device.py             188   Hardware detection + model recommendation
│   ├── engine.py             357   Core engine (3-phase pipeline + callbacks)
│   ├── profiles.py           254   Model-specific architecture parameters
│   ├── utils.py               94   Hashing, similarity, content extraction
│   ├── video.py              553   StreamCapture + load + dedup + motion gate
│   ├── tome.py               161   ToMe bipartite soft matching algorithm
│   ├── tome_vision.py        371   ViT wrappers (Qwen2.5-VL + Qwen3-VL)
│   ├── tome_backend.py       308   MLX backend with ToMe integration
│   ├── eval.py               406   Synthetic eval framework
│   ├── eval_benchmarks.py    491   POPE/TextVQA/Custom benchmarks
│   ├── token_compression.py  206   Post-encoder compression (baseline)
│   ├── compressed_backend.py 264   Compressed backend with custom generate
│   └── api/
│       ├── models.py         152   Pydantic request/response
│       └── server.py         245   FastAPI server
├── research/                       Research docs + eval results
│   ├── README.md
│   ├── visual-token-compression.md
│   ├── tome-implementation-plan.md
│   └── eval-results/
└── tests/                          120 tests
    ├── test_api.py           114
    ├── test_backends.py       61
    ├── test_callbacks.py      59
    ├── test_device.py         50
    ├── test_engine.py        139
    ├── test_profiles.py       70
    ├── test_stream_capture.py 99
    ├── test_tome.py          275   ToMe algorithm unit tests
    ├── test_tome_integration.py 296   Backend + wrapper integration tests
    └── test_video.py          93
```

## Roadmap

- [x] **v0.1:** Core engine — video pipeline, temporal dedup, motion gating
- [x] **v0.2:** Visual token compression (ToMe) + eval framework (POPE, TextVQA)
- [x] **v0.2.1:** Qwen3-VL support, EngineConfig integration, production hardening
- [ ] **Next:** Adaptive r per layer, KV cache interaction benchmarks
- [ ] **Phase 3:** VideoCache (KV cache reuse), batch scheduling
- [ ] **Phase 4:** `/metrics` endpoint, PyPI packaging

## License

Apache 2.0
