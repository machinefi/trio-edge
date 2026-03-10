# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Webcam GUI example with live video preview and VLM text overlay
- Accuracy regression test suite (POPE + TextVQA gate)
- Native engine plan and inference pipeline documentation

## [0.8.2] - 2026-03-10

### Added
- OpenClaw integration: `trio claw` connects to Gateway as a WebSocket node
- `vision.watch` + `vision.watch.stop` for continuous remote camera monitoring
- Ed25519 device identity for Gateway authentication
- `camera.snap`, `camera.list`, `vision.analyze`, `vision.status` commands
- Production hardening: graceful shutdown (SIGINT/SIGTERM), signal-based model reload (SIGHUP)
- Health endpoint with detailed status (`/health`)
- Structured JSON logging (`--json-logs` / `TRIO_LOG_JSON=1`)
- Request body size limit (10 MB) and resolution clamping (4K max)

### Fixed
- Watch loop reconnect on RTSP failure
- `<think>` tag stripping for Qwen3.5 models in all backends

## [0.8.1] - 2026-03-09

### Added
- SurveillanceVQA benchmark: 1,827 samples across 13 UCF-Crime anomaly categories
- LoRA fine-tuning pipeline (`scripts/prepare_surveillance_lora.py`)
- Shipped LoRA adapter: `adapters/surveillance-qwen35-2b/` (97.6% detection, POPE 89%)
- `--adapter-path` flag for bench.py and EngineConfig
- mlx-vlm 0.4.0 compatibility patches (4 bugs fixed)

### Changed
- bench.py auto-bumps `max_tokens` for thinking models on MCQ benchmarks
- BenchmarkRunner strips `<think>...</think>` tags from model output

## [0.8.0] - 2026-03-07

### Added
- ModelAdapter abstraction: per-architecture dispatch for token IDs, merge, RoPE, vision encoder
- 5 adapters: Qwen2.5-VL, Qwen3-VL, Qwen3.5, InternVL3, nanoLLaVA
- InternVL3 (1B/2B) promoted to Tier 1 with Compressed backend support
- nanoLLaVA promoted to Tier 1 with ToMe support (SigLIP ViT)
- Unified benchmark framework: `bench.py` + `examples/run_bench.py` with 12 models, sweep configs, anomaly detection
- 5 benchmark suites: POPE (100), TextVQA (50), GQA (50), MMBench (50), MVBench (12x5)
- `--skip-existing`, `--dry-run`, `--report` modes for benchmark sweeps

### Fixed
- Compressed/ToMe backends: proper guards for non-Qwen architectures
- bench.py: `auto_optimize=False` to prevent baseline pollution

### Results
- InternVL3-2B: POPE 95%, MMBench 98% (new best)
- Compressed 50% on 4B+ models acts as regularization, often improving accuracy
- FastV: removed (benchmark-negative on all T1 models)

## [0.2.1] - 2026-03-06

### Added
- Qwen3.5 support (0.8B / 4B / 9B profiles, verified ToMe compatibility)
- Gemma 3 model profiles (4B, 12B) with SigLIP vision encoder parameters
- SmolVLM model profiles (256M, 500M, 2.2B) for ultra-lightweight inference
- Pattern matching for HuggingFace model IDs of all supported families

### Changed
- Generalized Transformers backend: replaced `Qwen2_5_VLForConditionalGeneration` with `AutoModelForVision2Seq` for universal VLM support
- Added generic PIL-based input path for non-Qwen models (Gemma 3, SmolVLM, etc.)

## [0.2.0] - 2026-03-04

### Added
- ToMe (Token Merging) visual token compression inside ViT blocks
- Windowed-attention-aware merging for Qwen2.5-VL
- Qwen3-VL support with deepstack feature handling
- Compressed position encoding after token merging
- POPE and TextVQA benchmark framework
- Synthetic eval framework (prefill, decode, memory profiling)
- Benchmark CLI with A/B comparison (`run_benchmark.py`, `run_eval.py`)

### Results
- Qwen2.5-VL-3B at 1080p: **73% prefill speedup**, 68% token reduction with ToMe r=4
- Qwen3-VL-4B POPE: **zero quality loss** (91% accuracy) with 31% prefill speedup

## [0.1.0] - 2026-03-01

### Added
- Core inference engine with three-phase pipeline (preprocess, inference, postprocess)
- MLX backend for Apple Silicon (M1-M4) via mlx-vlm
- Transformers backend for NVIDIA GPU and CPU
- Hardware auto-detection and model recommendation
- StreamCapture for live video streams (RTSP, YouTube, webcam)
- Temporal deduplication (normalized L2 on 64x64 downscale)
- Motion gating (frame differencing + EMA background)
- Model profiles with architecture-aware parameters
- FastAPI server with OpenAI-compatible chat endpoint
- CLI: `trio-core serve`, `trio-core analyze`, `trio-core device`
- Callback system with 10 lifecycle events
- 120 unit tests

[Unreleased]: https://github.com/machinefi/trio-core/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/machinefi/trio-core/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/machinefi/trio-core/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/machinefi/trio-core/compare/v0.2.1...v0.8.0
[0.2.1]: https://github.com/machinefi/trio-core/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/machinefi/trio-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/machinefi/trio-core/releases/tag/v0.1.0
