<p align="center">
  <h1 align="center">TrioCore</h1>
  <p align="center">
    <strong>Portable video inference engine for Vision Language Models</strong>
  </p>
  <p align="center">
    Runs on Apple Silicon, NVIDIA GPU, and CPU — 2,540 lines, zero Docker.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#examples">Examples</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#api">API</a> |
  <a href="#model-profiles">Model Profiles</a>
</p>

---

## Why TrioCore?

Every VLM engine treats vision as "one image per request." TrioCore treats **video as a first-class input** with temporal deduplication, motion gating, and streaming capture.

**The thesis:** The biggest win in video VLM isn't faster inference — it's **calling the VLM less**. Temporal dedup + motion gating eliminates 50-90% of VLM calls on typical monitoring video.

```
Roboflow Inference:  "What objects are in this frame?"     (YOLO at 100fps)
TrioCore:          "What is happening in this video?"    (Qwen-VL at 2fps, skip 80%)
```

| | mlx-vlm | vllm-mlx | Roboflow Inference | **TrioCore** |
|---|---|---|---|---|
| Lines | 66K | 37K | 370K | **2,540** |
| Video-first | Partial | No | No | **Yes** |
| Stream capture | No | No | No | **Yes** |
| Temporal dedup | No | No | Coarse | **Yes** |
| Motion gating | No | No | No | **Yes** |
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
engine.load()  # auto-detects: M3 → MLX, RTX 4090 → Transformers

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
| [`webcam_to_text.py`](examples/webcam_to_text.py) | Live laptop camera → continuous VLM analysis |
| [`video_analyze.py`](examples/video_analyze.py) | Analyze a video file with detailed metrics |
| [`stream_monitor.py`](examples/stream_monitor.py) | Monitor RTSP/YouTube stream with motion gating |

```python
# Webcam → Text (3 lines of code)
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
   Model Profile          auto-select params for Qwen3.5/2.5-VL
        |
   Backend Auto-Select    detect_device() -> MLXBackend / TransformersBackend
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

## Model Profiles

TrioCore is optimized for the Qwen VL family. Each model has different architecture parameters — using the wrong ones wastes context or produces misaligned tensors.

```python
from trio_core import get_profile

profile = get_profile("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
print(profile.merge_factor)       # 28
print(profile.max_visual_tokens)  # 24576
```

| | Qwen3.5-0.8B | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|---|---|---|---|
| **Architecture** | Gated DeltaNet + Attn | Full GQA | Full GQA |
| **Context** | 262K | 128K | 128K |
| **Patch size** | 16 | 14 | 14 |
| **merge_factor** | 32 | 28 | 28 |
| **Max visual tokens** | 8,192 | 24,576 | 24,576 |
| **DeltaNet layers** | 18/24 | 0 | 0 |
| **KV heads** | 2 | 2 | 4 |
| **4-bit size** | ~0.5 GB | ~1.8 GB | ~4.5 GB |

The engine automatically selects the correct `merge_factor` for resize alignment and computes optimal `(frames, height, width)` to fit within each model's visual token budget:

```python
# Visual tokens = (frames / temporal_patch) x (H / merge_factor) x (W / merge_factor)
profile.compute_visual_tokens(8, 224, 224)  # 256 tokens

# Auto-optimize dimensions for token budget
frames, h, w = profile.compute_optimal_params(
    duration_sec=30.0, native_height=1080, native_width=1920
)
```

## Key Technologies

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
    max_tokens=1024,
)
engine = TrioCore(config)
```

## Project Structure

```
trio-core/                        2,540 lines production code
├── pyproject.toml
├── examples/
│   ├── webcam_to_text.py           Laptop camera → continuous VLM
│   ├── video_analyze.py            File analysis with metrics
│   └── stream_monitor.py           Live stream monitoring
├── src/trio_core/
│   ├── __init__.py            32   Public API exports
│   ├── backends.py           420   BaseBackend + MLXBackend + TransformersBackend
│   ├── callbacks.py           80   Hook system (10 events)
│   ├── cli.py                172   serve / analyze / bench / device
│   ├── config.py              50   Pydantic settings (TRIO_ env prefix)
│   ├── device.py             188   Hardware detection + model recommendation
│   ├── engine.py             334   Core engine (3-phase pipeline + callbacks)
│   ├── profiles.py           232   Model-specific architecture parameters
│   ├── utils.py               94   Hashing, similarity, content extraction
│   ├── video.py              541   StreamCapture + load + dedup + motion gate
│   └── api/
│       ├── models.py         152   Pydantic request/response
│       └── server.py         245   FastAPI server
└── tests/                          72 tests, all passing
    ├── test_api.py           114
    ├── test_backends.py       61
    ├── test_callbacks.py      59
    ├── test_device.py         50
    ├── test_engine.py        139
    ├── test_profiles.py       70
    ├── test_stream_capture.py 99
    └── test_video.py          93
```

## Roadmap

- **Phase 2:** VideoCache (3-tier prefix cache), KV cache reuse for repeated prompts
- **Phase 3:** Benchmarks, `/metrics` endpoint, PyPI packaging

## License

Apache 2.0
