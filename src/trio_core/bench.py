"""Unified benchmark infrastructure for TrioCore.

Provides a single extensible framework for running VLM benchmarks across
models, optimizations, and datasets. Adding a model/benchmark/optimization
is a one-liner in the respective registry.

Usage:
    from trio_core.bench import BenchSweep, get_models, OPTIM_PRESETS

    sweep = BenchSweep(
        models=get_models(families=["qwen2.5-vl"]),
        benchmarks=["pope"],
        configs=["baseline", "tome_r4"],
        max_samples=50,
    )
    sweep.plan()   # preview
    sweep.run()    # execute
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from trio_core.profiles import ModelProfile, get_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Model Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """Single source of truth for a benchmarkable model."""
    key: str
    hf_id: str
    profile: ModelProfile
    tier: int = 1
    _supports_fastv: bool = True

    @property
    def supports_tome(self) -> bool:
        return self.profile.supports_tome

    @property
    def supports_fastv(self) -> bool:
        return self._supports_fastv

    @property
    def supports_kv_reuse(self) -> bool:
        return True

    @property
    def supports_streammem(self) -> bool:
        return True

    @property
    def family(self) -> str:
        return self.profile.family


def _entry(key: str, hf_id: str, *, tier: int = 1, supports_fastv: bool = True) -> ModelEntry:
    return ModelEntry(
        key=key, hf_id=hf_id, profile=get_profile(hf_id),
        tier=tier, _supports_fastv=supports_fastv,
    )


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Qwen2.5-VL
    "qwen2.5-vl-3b": _entry("qwen2.5-vl-3b", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"),
    "qwen2.5-vl-7b": _entry("qwen2.5-vl-7b", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", supports_fastv=False),
    # Qwen3-VL
    "qwen3-vl-2b": _entry("qwen3-vl-2b", "mlx-community/Qwen3-VL-2B-Instruct-4bit"),
    "qwen3-vl-4b": _entry("qwen3-vl-4b", "mlx-community/Qwen3-VL-4B-Instruct-4bit"),
    "qwen3-vl-8b": _entry("qwen3-vl-8b", "mlx-community/Qwen3-VL-8B-Instruct-4bit"),
    # Qwen3.5 (DeltaNet hybrid) — FastV incompatible (DeltaNet layers have no self_attn)
    "qwen3.5-0.8b": _entry("qwen3.5-0.8b", "mlx-community/Qwen3.5-0.8B-MLX-4bit", supports_fastv=False),
    "qwen3.5-2b": _entry("qwen3.5-2b", "mlx-community/Qwen3.5-2B-4bit", supports_fastv=False),
    "qwen3.5-4b": _entry("qwen3.5-4b", "mlx-community/Qwen3.5-4B-MLX-4bit", supports_fastv=False),
    "qwen3.5-9b": _entry("qwen3.5-9b", "mlx-community/Qwen3.5-9B-MLX-4bit", supports_fastv=False),
    # InternVL3 — FastV produces garbage, ToMe incompatible (pixel_shuffle)
    "internvl3-1b": _entry("internvl3-1b", "mlx-community/InternVL3-1B-4bit", supports_fastv=False),
    "internvl3-2b": _entry("internvl3-2b", "mlx-community/InternVL3-2B-4bit", supports_fastv=False),
    # FastVLM — CoreML blocker, Tier 2
    "fastvlm-0.5b": _entry("fastvlm-0.5b", "InsightKeeper/FastVLM-0.5B-MLX-4bit", tier=2, supports_fastv=False),
    "fastvlm-1.5b": _entry("fastvlm-1.5b", "InsightKeeper/FastVLM-1.5B-MLX-4bit", tier=2, supports_fastv=False),
    # nanoLLaVA — demoted to T2 (upstream mlx_vlm issues, 0% POPE accuracy)
    "nanollava-1.5": _entry("nanollava-1.5", "mlx-community/nanoLLaVA-1.5-4bit", tier=2),
}


def get_models(
    keys: list[str] | None = None,
    families: list[str] | None = None,
    tier: int | None = None,
) -> list[ModelEntry]:
    """Filter models from the registry."""
    models = list(MODEL_REGISTRY.values())
    if keys is not None:
        models = [m for m in models if m.key in keys]
    if families is not None:
        models = [m for m in models if m.family in families]
    if tier is not None:
        models = [m for m in models if m.tier == tier]
    return models


# ---------------------------------------------------------------------------
# 2. Optimization Presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimConfig:
    """Named optimization configuration with capability requirements."""
    name: str
    description: str
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()


OPTIM_PRESETS: dict[str, OptimConfig] = {
    "baseline": OptimConfig(
        name="baseline",
        description="TrioCore default backend (no optimizations)",
        engine_kwargs={},
    ),
    "mlxvlm_baseline": OptimConfig(
        name="mlxvlm_baseline",
        description="Raw mlx-vlm generate (bypass TrioCore)",
        engine_kwargs={"_mlxvlm_baseline": True},
    ),
    "tome_r4": OptimConfig(
        name="tome_r4",
        description="Token Merging r=4",
        engine_kwargs={"tome_enabled": True, "tome_r": 4, "tome_metric": "hidden", "tome_min_keep_ratio": 0.3},
        requires=("supports_tome",),
    ),
    "tome_r8": OptimConfig(
        name="tome_r8",
        description="Token Merging r=8",
        engine_kwargs={"tome_enabled": True, "tome_r": 8, "tome_metric": "hidden", "tome_min_keep_ratio": 0.3},
        requires=("supports_tome",),
    ),
    "fastv": OptimConfig(
        name="fastv",
        description="FastV attention pruning (ratio=0.5, layer=2)",
        engine_kwargs={"fastv_enabled": True, "fastv_ratio": 0.5, "fastv_layer": 2},
        requires=("supports_fastv",),
    ),
    "fastv_r30": OptimConfig(
        name="fastv_r30",
        description="FastV attention pruning (ratio=0.3, layer=2)",
        engine_kwargs={"fastv_enabled": True, "fastv_ratio": 0.3, "fastv_layer": 2},
        requires=("supports_fastv",),
    ),
    "tome_fastv": OptimConfig(
        name="tome_fastv",
        description="ToMe r=4 + FastV combined",
        engine_kwargs={
            "tome_enabled": True, "tome_r": 4, "tome_metric": "hidden", "tome_min_keep_ratio": 0.3,
            "fastv_enabled": True, "fastv_ratio": 0.5, "fastv_layer": 2,
        },
        requires=("supports_tome", "supports_fastv"),
    ),
    "compressed_30": OptimConfig(
        name="compressed_30",
        description="Post-encoder token compression 30%",
        engine_kwargs={"_compress": 0.3},
    ),
    "compressed_40": OptimConfig(
        name="compressed_40",
        description="Post-encoder token compression 40%",
        engine_kwargs={"_compress": 0.4},
    ),
    "compressed_50": OptimConfig(
        name="compressed_50",
        description="Post-encoder token compression 50%",
        engine_kwargs={"_compress": 0.5},
    ),
    "compressed_60": OptimConfig(
        name="compressed_60",
        description="Post-encoder token compression 60%",
        engine_kwargs={"_compress": 0.6},
    ),
    "tome_compressed_50": OptimConfig(
        name="tome_compressed_50",
        description="ToMe r=4 + Compressed 50%",
        engine_kwargs={
            "tome_enabled": True, "tome_r": 4, "tome_metric": "hidden", "tome_min_keep_ratio": 0.3,
            "_compress": 0.5,
        },
        requires=("supports_tome",),
    ),
    "tome_compressed_40": OptimConfig(
        name="tome_compressed_40",
        description="ToMe r=4 + Compressed 40%",
        engine_kwargs={
            "tome_enabled": True, "tome_r": 4, "tome_metric": "hidden", "tome_min_keep_ratio": 0.3,
            "_compress": 0.4,
        },
        requires=("supports_tome",),
    ),
    "kv_reuse": OptimConfig(
        name="kv_reuse",
        description="Visual similarity KV reuse (threshold=0.95)",
        engine_kwargs={"visual_similarity_threshold": 0.95},
        requires=("supports_kv_reuse",),
    ),
    "streammem": OptimConfig(
        name="streammem",
        description="Streaming memory (budget=6000)",
        engine_kwargs={"streaming_memory_enabled": True, "streaming_memory_budget": 6000},
        requires=("supports_streammem",),
    ),
}


def is_compatible(model: ModelEntry, config: OptimConfig) -> bool:
    """Check if a model supports all capabilities required by a config."""
    for req in config.requires:
        if not getattr(model, req, False):
            return False
    return True


# ---------------------------------------------------------------------------
# 3. Benchmark Factory
# ---------------------------------------------------------------------------

def get_benchmark(name: str, max_samples: int | None = None, **kwargs):
    """Create a benchmark instance by name.

    Supported: pope, textvqa, gqa, mmbench, mvbench
    """
    from trio_core.eval_benchmarks import (
        POPEBenchmark, TextVQABenchmark, GQABenchmark,
        MMBenchBenchmark, MVBenchBenchmark,
    )

    factories = {
        "pope": lambda: POPEBenchmark(
            split=kwargs.get("split", "random"),
            max_samples=max_samples,
        ),
        "textvqa": lambda: TextVQABenchmark(max_samples=max_samples),
        "gqa": lambda: GQABenchmark(max_samples=max_samples),
        "mmbench": lambda: MMBenchBenchmark(max_samples=max_samples),
        "mvbench": lambda: MVBenchBenchmark(
            video_dir=kwargs.get("mvbench_dir", "./mvbench_videos"),
            tasks=kwargs.get("mvbench_tasks"),
            max_samples_per_task=max_samples,
            max_frames=kwargs.get("max_frames", 16),
        ),
    }

    if name not in factories:
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {list(factories)}")
    return factories[name]()


BENCHMARKS = ["pope", "textvqa", "gqa", "mmbench", "mvbench"]

# ---------------------------------------------------------------------------
# 4. Anomaly Detector
# ---------------------------------------------------------------------------

def detect_anomalies(result, benchmark_name: str) -> list[str]:
    """Flag suspicious benchmark results."""
    anomalies = []
    acc = result.accuracy

    if acc == 0.0:
        anomalies.append("0% accuracy — model producing garbage")
    elif benchmark_name == "pope" and acc < 0.40:
        anomalies.append(f"POPE accuracy {acc:.1%} < 40% — likely broken")

    if benchmark_name == "pope":
        yr = result.yes_rate
        if yr > 0.85:
            anomalies.append(f"POPE yes_rate {yr:.1%} > 85% — heavy YES bias")
        elif yr < 0.15:
            anomalies.append(f"POPE yes_rate {yr:.1%} < 15% — heavy NO bias")

    if result.avg_latency_ms > 30000:
        anomalies.append(f"avg_latency {result.avg_latency_ms:.0f}ms > 30s — abnormally slow")

    if result.avg_prompt_tokens == 0:
        anomalies.append("prompt_tokens = 0 — metrics not collected")

    return anomalies


# ---------------------------------------------------------------------------
# 5. Result Store
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("research/bench-results")


@dataclass
class RunRecord:
    """Denormalized record of a single benchmark run."""
    model_key: str
    benchmark: str
    config: str
    timestamp: str
    file_path: str
    accuracy: float
    f1: float
    avg_latency_ms: float
    n_samples: int
    anomalies: list[str] = field(default_factory=list)
    yes_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResultStore:
    """Structured directory + JSON manifest for benchmark results."""

    def __init__(self, base_dir: Path | str = RESULTS_DIR):
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "manifest.json"
        self._manifest: list[dict] | None = None

    def _load_manifest(self) -> list[dict]:
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())
        else:
            self._manifest = []
        return self._manifest

    def _save_manifest(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._load_manifest(), indent=2))

    def save(self, model_key: str, benchmark: str, config: str, result) -> Path:
        """Save a BenchmarkResult and add to manifest."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        model_dir = self.base_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{benchmark}__{config}__{ts}.json"
        filepath = model_dir / filename

        result.save(str(filepath))

        anomalies = detect_anomalies(result, benchmark)

        record = {
            "model_key": model_key,
            "benchmark": benchmark,
            "config": config,
            "timestamp": ts,
            "file_path": str(filepath.relative_to(self.base_dir)),
            "accuracy": result.accuracy,
            "f1": result.f1,
            "avg_latency_ms": result.avg_latency_ms,
            "n_samples": result.n_samples,
            "anomalies": anomalies,
            "yes_rate": getattr(result, "yes_rate", None),
        }

        manifest = self._load_manifest()
        manifest.append(record)
        self._save_manifest()

        if anomalies:
            print(f"  ANOMALIES: {'; '.join(anomalies)}")

        return filepath

    def query(
        self,
        model_key: str | None = None,
        benchmark: str | None = None,
        config: str | None = None,
    ) -> list[RunRecord]:
        """Query manifest for matching records."""
        records = []
        for entry in self._load_manifest():
            if model_key and entry["model_key"] != model_key:
                continue
            if benchmark and entry["benchmark"] != benchmark:
                continue
            if config and entry["config"] != config:
                continue
            records.append(RunRecord(**{
                k: entry[k] for k in RunRecord.__dataclass_fields__ if k in entry
            }))
        return records

    def latest(self, model_key: str, benchmark: str, config: str) -> RunRecord | None:
        """Get the most recent record for a specific combo."""
        matches = self.query(model_key=model_key, benchmark=benchmark, config=config)
        return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# 6. Engine Factory
# ---------------------------------------------------------------------------

def build_engine(model: ModelEntry, config: OptimConfig, *, max_tokens: int = 16):
    """Build and load an engine for benchmarking.

    Returns (engine_or_backend, is_mlxvlm_baseline).

    Handles three modes:
    - TrioCore (default): standard engine with optimization config
    - CompressedMLXBackend: when _compress is in engine_kwargs
    - mlx-vlm baseline: raw mlx_vlm.generate (bypass TrioCore)
    """
    from trio_core import TrioCore
    from trio_core.config import EngineConfig

    kwargs = config.engine_kwargs
    is_mlxvlm = kwargs.get("_mlxvlm_baseline", False)
    is_compress = "_compress" in kwargs

    if is_mlxvlm:
        # Raw mlx-vlm baseline — return (model, processor)
        from mlx_vlm import load
        vlm_model, processor = load(model.hf_id, trust_remote_code=True)
        return (vlm_model, processor), True

    # Build EngineConfig with non-special kwargs
    config_kwargs = {
        "model": model.hf_id,
        "max_tokens": max_tokens,
        "dedup_enabled": False,
        "motion_enabled": False,
    }
    for k, v in kwargs.items():
        if not k.startswith("_"):
            config_kwargs[k] = v

    engine_config = EngineConfig(**config_kwargs)
    engine = TrioCore(engine_config)

    if is_compress:
        from trio_core.token_compression import TokenCompressor
        from trio_core.compressed_backend import CompressedMLXBackend
        ratio = kwargs["_compress"]
        compressor = TokenCompressor(strategy="similarity", ratio=ratio)
        backend = CompressedMLXBackend(model.hf_id, compressor)
        backend.load()

        # Apply ToMe wrapping if both compress + tome are requested
        if kwargs.get("tome_enabled", False):
            from trio_core.native_vision import create_tome_vision
            if backend._adapter and backend._adapter.supports_tome:
                native_vision = create_tome_vision(
                    backend._model.vision_tower,
                    tome_r=kwargs.get("tome_r", 4),
                    metric=kwargs.get("tome_metric", "hidden"),
                    min_keep_ratio=kwargs.get("tome_min_keep_ratio", 0.3),
                )
                backend._model.vision_tower = native_vision
                logger.info("Compound mode: ToMe (vision) + Compressed %.0f%%", ratio * 100)

        engine._backend = backend
        engine._loaded = True
    else:
        engine.load()

    return engine, False


def _run_mlxvlm_baseline(model_proc, benchmark, max_tokens: int = 16):
    """Run benchmark using raw mlx-vlm generate (for baseline comparison)."""
    import numpy as np
    from PIL import Image
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from trio_core.eval_benchmarks import BenchmarkResult, PredictionResult

    model, processor = model_proc
    samples = benchmark.load()
    predictions = []
    t0 = time.perf_counter()

    for i, sample in enumerate(samples):
        question = sample.question
        if hasattr(benchmark, "PROMPT_TEMPLATE"):
            question = benchmark.PROMPT_TEMPLATE.format(question=sample.question)

        # Convert frame to PIL
        frame = sample.image[0] if sample.image.ndim == 4 else sample.image
        if frame.dtype == np.float32:
            if frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(frame)

        # Build chat-template prompt (handles model-specific formatting)
        prompt = apply_chat_template(processor, model.config, question, num_images=1)

        tic = time.perf_counter()
        result = generate(
            model, processor, prompt, image=[pil_img],
            max_tokens=max_tokens, temperature=0.0, verbose=False,
        )
        elapsed = (time.perf_counter() - tic) * 1000

        output = result.text.strip() if hasattr(result, "text") else str(result).strip()
        correct = benchmark.judge(sample, output)
        predictions.append(PredictionResult(
            id=sample.id, question=sample.question,
            answer_gt=sample.answer, answer_pred=output,
            correct=correct, latency_ms=elapsed,
        ))

        if (i + 1) % 50 == 0:
            acc = sum(p.correct for p in predictions) / len(predictions)
            print(f"  [{i+1}/{len(samples)}] accuracy={acc:.1%}")

    total_time = time.perf_counter() - t0
    return BenchmarkResult(
        name=benchmark.name, n_samples=len(predictions),
        predictions=predictions, total_time_s=total_time,
        metadata={"backend": "mlx-vlm-baseline"},
    )


# ---------------------------------------------------------------------------
# 7. Sweep Orchestrator
# ---------------------------------------------------------------------------

class BenchSweep:
    """Run models x benchmarks x configs with auto-filtering."""

    def __init__(
        self,
        models: list[ModelEntry] | None = None,
        benchmarks: list[str] | None = None,
        configs: list[str] | None = None,
        max_samples: int | None = None,
        max_tokens: int = 16,
        skip_existing: bool = False,
        results_dir: Path | str = RESULTS_DIR,
        benchmark_kwargs: dict[str, Any] | None = None,
    ):
        self.models = models or get_models(tier=1)
        self.benchmark_names = benchmarks or ["pope"]
        self.config_names = configs or ["baseline"]
        self.max_samples = max_samples
        self.max_tokens = max_tokens
        self.skip_existing = skip_existing
        self.store = ResultStore(results_dir)
        self.benchmark_kwargs = benchmark_kwargs or {}

    def plan(self) -> list[tuple[ModelEntry, str, OptimConfig]]:
        """Preview what will run, auto-filtering incompatible combos."""
        combos = []
        skipped = []

        for model in self.models:
            for bench_name in self.benchmark_names:
                for config_name in self.config_names:
                    config = OPTIM_PRESETS.get(config_name)
                    if config is None:
                        print(f"  WARNING: unknown config {config_name!r}, skipping")
                        continue

                    if not is_compatible(model, config):
                        missing = [r for r in config.requires if not getattr(model, r, False)]
                        skipped.append((model.key, bench_name, config_name, missing))
                        continue

                    if self.skip_existing:
                        existing = self.store.latest(model.key, bench_name, config_name)
                        if existing:
                            skipped.append((model.key, bench_name, config_name, ["already exists"]))
                            continue

                    combos.append((model, bench_name, config))

        # Print plan
        print(f"\n{'='*60}")
        print(f"  BENCH SWEEP PLAN")
        print(f"{'='*60}")
        print(f"  Models:     {len(self.models)}")
        print(f"  Benchmarks: {self.benchmark_names}")
        print(f"  Configs:    {self.config_names}")
        print(f"  Samples:    {self.max_samples or 'all'}")
        print(f"  Combos:     {len(combos)} to run, {len(skipped)} skipped")

        if skipped:
            print(f"\n  Skipped:")
            for model_key, bench, cfg, reasons in skipped:
                print(f"    {model_key} / {bench} / {cfg}: {', '.join(reasons)}")

        print(f"\n  Will run:")
        for model, bench_name, config in combos:
            print(f"    {model.key} / {bench_name} / {config.name}")

        print()
        return combos

    def run(self) -> list[RunRecord]:
        """Execute the sweep. Groups by (model, config) to minimize reloads."""
        combos = self.plan()
        if not combos:
            print("  Nothing to run.")
            return []

        # Group by (model_key, config_name) to avoid reloading
        from collections import OrderedDict
        groups: OrderedDict[tuple[str, str], list[tuple[ModelEntry, str, OptimConfig]]] = OrderedDict()
        # Sort by model key first to group model loads
        for model, bench_name, config in sorted(combos, key=lambda x: (x[0].key, x[2].name)):
            group_key = (model.key, config.name)
            groups.setdefault(group_key, []).append((model, bench_name, config))

        records: list[RunRecord] = []
        total = len(combos)
        done = 0

        for (model_key, config_name), group_combos in groups.items():
            model = group_combos[0][0]
            config = group_combos[0][2]

            print(f"\n{'#'*60}")
            print(f"#  Loading: {model.key} / {config.name}")
            print(f"#  HF ID:  {model.hf_id}")
            print(f"{'#'*60}")

            try:
                tic = time.perf_counter()
                engine_or_backend, is_mlxvlm = build_engine(model, config, max_tokens=self.max_tokens)
                load_time = time.perf_counter() - tic
                print(f"  Loaded in {load_time:.1f}s")

                for _, bench_name, _ in group_combos:
                    done += 1
                    print(f"\n  [{done}/{total}] {model.key} / {bench_name} / {config.name}")

                    try:
                        benchmark = get_benchmark(bench_name, max_samples=self.max_samples, **self.benchmark_kwargs)

                        if is_mlxvlm:
                            result = _run_mlxvlm_baseline(engine_or_backend, benchmark, max_tokens=self.max_tokens)
                        else:
                            from trio_core.eval_benchmarks import BenchmarkRunner
                            runner = BenchmarkRunner(engine_or_backend, max_tokens=self.max_tokens)
                            result = runner.run(benchmark)

                        result.print()

                        filepath = self.store.save(model.key, bench_name, config.name, result)
                        print(f"  Saved: {filepath}")

                        record = self.store.latest(model.key, bench_name, config.name)
                        if record:
                            records.append(record)

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()

                # Cleanup
                del engine_or_backend
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except ImportError:
                    pass

            except Exception as e:
                print(f"  LOAD ERROR: {e}")
                import traceback
                traceback.print_exc()
                done += len(group_combos)

        return records


# ---------------------------------------------------------------------------
# 8. Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate comparison tables from benchmark results."""

    def __init__(self, store: ResultStore | None = None):
        self.store = store or ResultStore()

    def model_x_config(self, benchmark: str, metric: str = "accuracy") -> str:
        """Models (rows) x Configs (cols) table."""
        records = self.store.query(benchmark=benchmark)
        if not records:
            return f"No results for benchmark={benchmark}"

        # Collect unique models and configs
        models = list(dict.fromkeys(r.model_key for r in records))
        configs = list(dict.fromkeys(r.config for r in records))

        # Build lookup (use latest per combo)
        lookup: dict[tuple[str, str], RunRecord] = {}
        for r in records:
            lookup[(r.model_key, r.config)] = r

        # Format table
        col_w = max(12, max((len(c) for c in configs), default=12))
        header = f"{'Model':<20} " + " ".join(f"{c:>{col_w}}" for c in configs)
        sep = "-" * len(header)
        lines = [f"\n{benchmark.upper()} — {metric}", sep, header, sep]

        for model in models:
            row = f"{model:<20} "
            for cfg in configs:
                rec = lookup.get((model, cfg))
                if rec is None:
                    row += f"{'—':>{col_w}} "
                else:
                    val = getattr(rec, metric, None)
                    if val is None:
                        row += f"{'—':>{col_w}} "
                    elif metric == "accuracy":
                        row += f"{val*100:>{col_w-1}.1f}% "
                    elif metric == "avg_latency_ms":
                        row += f"{val:>{col_w-2}.0f}ms "
                    else:
                        row += f"{val:>{col_w}.3f} "
            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def model_x_benchmark(self, config: str = "baseline", metric: str = "accuracy") -> str:
        """Models (rows) x Benchmarks (cols) table."""
        records = self.store.query(config=config)
        if not records:
            return f"No results for config={config}"

        models = list(dict.fromkeys(r.model_key for r in records))
        benchmarks = list(dict.fromkeys(r.benchmark for r in records))

        lookup: dict[tuple[str, str], RunRecord] = {}
        for r in records:
            lookup[(r.model_key, r.benchmark)] = r

        col_w = max(12, max((len(b) for b in benchmarks), default=12))
        header = f"{'Model':<20} " + " ".join(f"{b:>{col_w}}" for b in benchmarks)
        sep = "-" * len(header)
        lines = [f"\nConfig: {config} — {metric}", sep, header, sep]

        for model in models:
            row = f"{model:<20} "
            for bench in benchmarks:
                rec = lookup.get((model, bench))
                if rec is None:
                    row += f"{'—':>{col_w}} "
                elif metric == "accuracy":
                    row += f"{rec.accuracy*100:>{col_w-1}.1f}% "
                else:
                    val = getattr(rec, metric, None)
                    row += f"{val:>{col_w}.3f} " if val else f"{'—':>{col_w}} "
            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def delta_report(self, benchmark: str, baseline: str, compare: str) -> str:
        """Show accuracy delta between two configs."""
        bl_records = {r.model_key: r for r in self.store.query(benchmark=benchmark, config=baseline)}
        cmp_records = {r.model_key: r for r in self.store.query(benchmark=benchmark, config=compare)}

        models = list(dict.fromkeys(list(bl_records) + list(cmp_records)))
        if not models:
            return f"No results for {benchmark} with configs {baseline}, {compare}"

        header = f"{'Model':<20} {'Baseline':>10} {'Compare':>10} {'Delta':>10}"
        sep = "-" * len(header)
        lines = [f"\n{benchmark.upper()} — {baseline} vs {compare}", sep, header, sep]

        for model in models:
            bl = bl_records.get(model)
            cmp = cmp_records.get(model)
            bl_str = f"{bl.accuracy*100:.1f}%" if bl else "—"
            cmp_str = f"{cmp.accuracy*100:.1f}%" if cmp else "—"
            if bl and cmp:
                delta = (cmp.accuracy - bl.accuracy) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.1f}%"
            else:
                delta_str = "—"
            lines.append(f"{model:<20} {bl_str:>10} {cmp_str:>10} {delta_str:>10}")

        lines.append(sep)
        return "\n".join(lines)
