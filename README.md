<p align="center">
  <h1 align="center">TrioCore</h1>
  <p align="center">
    <strong>Video-first inference engine for Vision Language Models on Apple Silicon</strong>
  </p>
  <p align="center">
    4 optimization stages. 12 Tier-1 models. 13K lines. Zero Docker.
  </p>
</p>

<p align="center">
  <a href="#optimization-pipeline">Optimization Pipeline</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#supported-models">Supported Models</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#api">API</a>
</p>

---

## Why TrioCore?

Every VLM engine treats vision as "one image per request." TrioCore treats **video as a first-class input** — and optimizes every stage of the inference pipeline.

**The thesis:** The biggest win in video VLM isn't faster decoding — it's **fewer visual tokens and fewer VLM calls**. Prefill cost is O(n^2) in token count. KV cache is O(n). Reducing tokens attacks both simultaneously.

## Optimization Pipeline

```
Input --> [Vision Encoder] --> visual tokens --> [LLM Prefill] --> [KV Cache] --> [Decode]
             ToMe                FastV              KV Reuse        StreamMem
          merge tokens        prune tokens      reuse across       bounded KV
          inside ViT          inside LLM          frames           for streams
```

TrioCore embeds **4 optimization techniques** at different stages of VLM inference. Each is independent and compounds with the others:

### Stage 1: ToMe -- Vision Encoder Token Merging

Training-free token merging inside the vision encoder, based on [Token Merging (ICLR 2023)](https://arxiv.org/abs/2210.09461). Bipartite soft matching merges similar visual tokens between ViT blocks, gradually reducing sequence length before it reaches the LLM.

| Technique | Where | What |
|---|---|---|
| Bipartite matching | Per ViT block | Merge r most-similar token pairs per block |
| Hidden-state metric | Similarity | Cosine similarity on intermediate features (not K-matrix) |
| Windowed-aware | Qwen2.5-VL | Merge only within attention windows |
| Adaptive r | Per image | Content-aware: complex images keep more tokens |
| Layer-adaptive ramp | Per block | Early blocks merge less, later blocks merge more |

**Results (M3 Pro, 4-bit):**

| Model | Resolution | Baseline Tokens | ToMe Tokens | Prefill Speedup | Accuracy |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 1080p | 748 | 242 (-68%) | **-73%** (1808->490ms) | 81% POPE |
| Qwen3-VL-4B | 480p | 278 | 258 (-7%) | **-31%** (835->573ms) | **91% POPE (zero loss)** |

### Stage 2: FastV -- Mid-Stream Visual Token Pruning

Attention-based importance scoring inside the LLM layers. After a designated "observation" layer, tokens with low attention from the `[CLS]`/text query are pruned from KV cache mid-forward-pass.

| Technique | Where | What |
|---|---|---|
| Attention scoring | LLM layer N | Score visual tokens by attention received from text tokens |
| Single-pass KV prune | Mid-prefill | Remove low-importance KV entries in-place (no recomputation) |
| MRoPE-aware | Position IDs | Recompute 3D position IDs after pruning to maintain spatial coherence |

### Stage 3: Frame-to-Frame KV Reuse

For video inference, consecutive frames share 80%+ visual content. Instead of full prefill per frame, TrioCore detects visual similarity and reuses the previous frame's KV cache.

| Technique | Where | What |
|---|---|---|
| Embedding similarity | Pre-prefill | Cosine similarity between current and cached visual embeddings |
| Four-tier cache | Hit logic | Exact match > visual similarity > text prefix > full miss |
| DeltaNet state | Qwen3.5 | Snapshot/restore recurrent state (not KV cache) for hybrid models |
| Input-ids guard | Safety | Reject reuse if prompt changed (prevents wrong-answer contamination) |

**Results (480p, 5-frame video sequences, threshold=0.95):**

| Model | Warm Speedup | Architecture |
|---|---|---|
| Qwen2.5-VL-3B | **1.57x** | KV cache trim + reuse |
| Qwen3-VL-4B | **1.71x** | KV cache trim + reuse |
| Qwen3.5-0.8B | **1.35x** | DeltaNet state snapshot/restore |

### Stage 4: StreamMem -- Bounded KV Cache for Streams

For long or continuous video, KV cache grows without bound. StreamMem caps memory via saliency-based eviction + prototype merging + attention sink.

| Technique | Where | What |
|---|---|---|
| Saliency scoring | Per eviction | Score KV entries by accumulated attention (proxy query from chat template) |
| Prototype merging | Eviction | Merge similar evicted entries rather than discarding |
| Attention sink | Protection | First N visual tokens immune to eviction (StreamingLLM-inspired) |
| Budget guard | Memory | Hard cap on total KV entries, triggers eviction when exceeded |

### Pre-Inference: Video Pipeline

Before any model runs, TrioCore reduces work at the frame level:

| Technique | Effect |
|---|---|
| Temporal dedup | **30-70% fewer frames** — skip near-identical consecutive frames (L2 on 64x64 grayscale) |
| Motion gate | **80%+ fewer VLM calls** — skip inference entirely when scene is static |
| StreamCapture | Daemon thread, webcam/RTSP/YouTube, grab/retrieve frame skipping |

### Compound Effect

All 4 optimization stages are independent and stack:

```
                        Baseline         With TrioCore
                        (raw VLM)        (all optimizations)
                        ---------        ------------------
VLM calls per video     every frame      motion gate: -80%+ calls
Visual tokens           748 (1080p)      ToMe: 242 (-68%)
Prefill latency         1,808ms          490ms (-73%)
Frame-to-frame          full prefill     KV reuse: 1.57-1.71x speedup
KV memory               unbounded        StreamMem: budget-capped
```

## Benchmarks

All benchmarks on Apple M3 Pro, 4-bit quantized models.

### POPE -- Object Hallucination (100 COCO images)

| Model | Config | Accuracy | F1 | Avg Latency | Visual Tokens |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | Baseline | 92.0% | 0.913 | 952ms | 363 |
| Qwen2.5-VL-3B | ToMe r=4 | 81.0% | 0.800 | 752ms (-21%) | 131 (-64%) |
| **Qwen3-VL-4B** | **Baseline** | **91.0%** | **0.901** | **795ms** | **278** |
| **Qwen3-VL-4B** | **ToMe r=4** | **91.0%** | **0.901** | **777ms (-2%)** | **258 (-7%)** |

Qwen3-VL shows **zero quality loss** with ToMe.

### Frame-to-Frame KV Reuse (480p, 5-frame sequences)

| Model | Warm Speedup | Warm Savings | Output Quality |
|---|---|---|---|
| Qwen2.5-VL-3B | **1.57x** | 36.4% | Identical |
| Qwen3-VL-4B | **1.71x** | 41.5% | Near-identical |
| Qwen3.5-0.8B | **1.35x** | 25.8% | Correct (DeltaNet) |

### Generate Loop Overhead (trio-core vs mlx-vlm, 480p, 32 tokens)

| Metric | mlx-vlm | trio-core | Diff |
|---|---|---|---|
| Prefill | 1018ms | 1016ms | -0.2% |
| Decode | 524ms | 513ms | -2.1% |
| Output | - | - | **5/5 bit-identical** |

Zero overhead from custom generate loop. Output bit-identical at temperature=0.

See [`research/`](research/) for detailed analysis, eval results, and implementation notes.

## Supported Models

### Tier 1 -- Full Optimization (12 models, 4 families)

All Tier 1 models have native model loading, ModelAdapter integration, and full access to all 4 optimization stages.

| Model | Params | 4-bit Size | ToMe | FastV | KV Reuse | StreamMem |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL-3B** | 3B | 1.8 GB | Yes | Yes | Yes | Yes |
| **Qwen2.5-VL-7B** | 7B | 4.5 GB | Yes | Yes | Yes | Yes |
| **Qwen3-VL-2B** | 2B | 1.5 GB | Yes | Yes | Yes | Yes |
| **Qwen3-VL-4B** | 4B | 2.5 GB | Yes | Yes | Yes | Yes |
| **Qwen3-VL-8B** | 8B | 5.0 GB | Yes | Yes | Yes | Yes |
| **Qwen3.5-0.8B** | 0.8B | 0.5 GB | Yes | Yes | Yes (DeltaNet) | Yes |
| **Qwen3.5-2B** | 2B | 1.5 GB | Yes | Yes | Yes (DeltaNet) | Yes |
| **Qwen3.5-4B** | 4B | 2.5 GB | Yes | Yes | Yes (DeltaNet) | Yes |
| **Qwen3.5-9B** | 9B | 5.0 GB | Yes | Yes | Yes (DeltaNet) | Yes |
| **InternVL3-1B** | 1B | 1.0 GB | No\* | Yes | Yes | Yes |
| **InternVL3-2B** | 2B | 1.6 GB | No\* | Yes | Yes | Yes |
| **nanoLLaVA-1.5** | 1B | 1.0 GB | Yes | Yes | Yes | Yes |

> LLM-side optimizations (FastV, KV Reuse, StreamMem) are model-agnostic.
> ViT-side optimization (ToMe) requires per-architecture wrapper.

\* InternVL3: pixel_shuffle after ViT disrupts spatial structure -- ToMe not applicable.

### Architecture Notes

| | Qwen2.5-VL | Qwen3-VL | Qwen3.5 |
|---|---|---|---|
| **ViT** | Qwen ViT (windowed attn) | Qwen ViT (full attn) | Qwen ViT (full attn) |
| **LLM** | Qwen2 (GQA) | Qwen2 (GQA) | DeltaNet + Attention (hybrid) |
| **RoPE** | 3D MRoPE (chunked) | 3D MRoPE (interleaved) | 3D MRoPE (interleaved) |
| **Deepstack** | No | Yes | Yes (empty) |
| **Patch size** | 14 | 14 | 16 |
| **Native loading** | Yes | Yes | Yes |

### Tier 2 -- Inference Only (mlx-vlm fallback, no optimization backends)

| Model | Family | Params | Notes |
|---|---|---|---|
| **Gemma 3n E2B/E4B** | gemma3n | 5B/8B | Edge-first, MatFormer |
| **SmolVLM2 2.2B** | smolvlm | 2.2B | Smallest practical VLM |
| **Phi-4 Multimodal** | phi4 | 3.8B | Strong STEM reasoning |
| **Gemma 3 4B/12B** | gemma3 | 4-12B | 140+ languages |
| **SmolVLM 256M** | smolvlm | 256M | Ultra-lightweight |
| **FastVLM 0.5B/1.5B** | fastvlm | 0.5-1.5B | CoreML vision encoder |

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
```

### Enable Optimizations

```python
from trio_core import TrioCore, EngineConfig

config = EngineConfig(
    tome_enabled=True,      # Stage 1: ToMe visual token merging
    tome_r=4,               # Merge 4 token pairs per ViT block
    tome_metric="hidden",   # Similarity metric (hidden > keys)
)
engine = TrioCore(config)
engine.load()
result = engine.analyze_video(frames, "What do you see?")
```

## Examples

See [`examples/`](examples/) for complete scripts:

| Example | Description |
|---|---|
| [`webcam_to_text.py`](examples/webcam_to_text.py) | Live laptop camera -> continuous VLM analysis |
| [`video_analyze.py`](examples/video_analyze.py) | Analyze a video file with detailed metrics |
| [`stream_monitor.py`](examples/stream_monitor.py) | Monitor RTSP/YouTube stream with motion gating |
| [`webcam_gui.py`](examples/webcam_gui.py) | Live camera + VLM overlay in GUI window |
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
   Model Profile          auto-select params per model family
        |
   Backend Auto-Select    detect_device() -> MLXBackend / TransformersBackend
        |
   [Stage 1] ToMe         bipartite soft matching inside ViT blocks
        |
   [Stage 2] FastV        attention-based visual token pruning mid-LLM
        |
   [Stage 3] KV Reuse     frame-to-frame visual similarity gating
        |
   [Stage 4] StreamMem    saliency eviction + prototype merging
        |
   VideoResult            text + InferenceMetrics (3-phase timing)
```

### Hardware Auto-Detection

| Hardware | Detection | Backend |
|---|---|---|
| Apple Silicon (M1-M4) | `arm64` + `sysctl` | MLX (native model loading, mlx-vlm fallback) |
| NVIDIA GPU | `nvidia-smi` / `torch.cuda` | Transformers (PyTorch) |
| CPU-only | Fallback | Transformers (PyTorch) |

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
TRIO_STREAMING_MEMORY_ENABLED=false
TRIO_STREAMING_MEMORY_BUDGET=6000
TRIO_STREAMING_MEMORY_PROTOTYPE_RATIO=0.1
TRIO_HOST=0.0.0.0
TRIO_PORT=8000
```

## Roadmap

- [x] **v0.1:** Core engine -- video pipeline, temporal dedup, motion gating
- [x] **v0.2:** Visual token compression (ToMe) + eval framework (POPE, TextVQA)
- [x] **v0.3:** Custom generate loop -- own KV cache, speculative decoding, early stopping
- [x] **v0.3.1:** Native ToMe ViT, FastV pruning, frame-to-frame KV reuse (1.7x speedup)
- [x] **v0.3.2:** Content-aware adaptive r, 4-tier cache hierarchy, StreamMem
- [x] **v0.3.3:** Native model loading -- vendored Qwen2.5-VL, Qwen3-VL, Qwen3.5 (zero mlx-vlm for T1)
- [ ] **v0.4:** Comprehensive benchmarks + Tier 2 model optimization
- [ ] **v0.5:** Full native engine -- zero mlx-vlm dependency
- [ ] **Ongoing:** Continuous batching, PyPI packaging

## License

Apache 2.0
