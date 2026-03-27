"""Eval framework — measure VLM inference efficiency and quality.

Metrics aligned with visual token compression papers:
  - NeurIPS 2024: "Efficient Large Multi-modal Models via Visual Context Compression"
  - ICLR 2025: "Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters"

Key metrics:
  - prompt_tokens:  visual + text token count (compression target)
  - prefill_ms:     prefill latency (∝ prompt_tokens², main speedup)
  - decode_ms:      decode latency (unchanged by compression)
  - peak_memory_gb: KV cache + activations (∝ prompt_tokens)
  - quality:        output text (compare before/after compression)

Usage:
    from trio_core.eval import EvalSuite
    suite = EvalSuite(engine)
    report = suite.run()
    report.print()
    report.save("baseline.json")

    # Compare two runs
    EvalReport.compare("baseline.json", "compressed.json")
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trio_core.engine import InferenceMetrics, TrioCore

# ── Test Case Generation ────────────────────────────────────────────────────


def generate_test_frames(
    complexity: str = "medium",
    height: int = 480,
    width: int = 640,
    n_frames: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic test frames with controlled visual complexity.

    Returns (T, C, H, W) float32 in [0, 1].
    """
    rng = np.random.RandomState(seed)

    if complexity == "solid":
        # Uniform color — minimal visual tokens needed
        frames = np.full((n_frames, 3, height, width), 0.3, dtype=np.float32)

    elif complexity == "gradient":
        # Smooth gradient — low complexity, high redundancy
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        grid = np.outer(y, x)  # (H, W)
        frame = np.stack([grid, 1 - grid, grid * 0.5], axis=0)  # (3, H, W)
        frames = np.stack([frame] * n_frames, axis=0)

    elif complexity == "medium":
        # Blocks of color — moderate complexity
        frame = np.zeros((3, height, width), dtype=np.float32)
        block_h, block_w = height // 8, width // 8
        for i in range(8):
            for j in range(8):
                color = rng.rand(3).astype(np.float32)
                frame[:, i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w] = color[
                    :, None, None
                ]
        frames = np.stack([frame] * n_frames, axis=0)

    elif complexity == "high":
        # Random noise — maximum complexity, no redundancy
        frames = rng.rand(n_frames, 3, height, width).astype(np.float32)

    elif complexity == "scene":
        # Simulated indoor scene — realistic token distribution
        frame = np.full((3, height, width), 0.9, dtype=np.float32)  # white wall
        # Floor
        frame[:, int(height * 0.7) :, :] = np.array([0.4, 0.35, 0.3])[:, None, None]
        # Window
        frame[:, int(height * 0.1) : int(height * 0.5), int(width * 0.6) : int(width * 0.9)] = (
            np.array([0.6, 0.8, 1.0])[:, None, None]
        )
        # Objects (colored rectangles)
        for _ in range(5):
            y1 = rng.randint(int(height * 0.3), int(height * 0.7))
            x1 = rng.randint(0, int(width * 0.7))
            h, w = rng.randint(30, 80), rng.randint(30, 80)
            color = rng.rand(3).astype(np.float32)
            frame[:, y1 : y1 + h, x1 : x1 + w] = color[:, None, None]
        frames = np.stack([frame] * n_frames, axis=0)

    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    return frames


# ── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """A single evaluation test case."""

    name: str
    frames: np.ndarray  # (T, C, H, W) float32
    prompt: str
    complexity: str = ""  # metadata: what kind of input


@dataclass
class RunMetrics:
    """Metrics from a single inference run."""

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Timing (ms)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    latency_ms: float = 0.0

    # Throughput
    prompt_tps: float = 0.0  # prefill tokens/sec
    generation_tps: float = 0.0  # decode tokens/sec

    # Memory
    peak_memory_gb: float = 0.0

    # Quality
    output_text: str = ""

    @staticmethod
    def from_result(metrics: InferenceMetrics, text: str) -> RunMetrics:
        return RunMetrics(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            prefill_ms=metrics.prefill_ms,
            decode_ms=metrics.decode_ms,
            inference_ms=metrics.inference_ms,
            latency_ms=metrics.latency_ms,
            prompt_tps=metrics.prompt_tps,
            generation_tps=metrics.tokens_per_sec,
            peak_memory_gb=metrics.peak_memory_gb,
            output_text=text,
        )


@dataclass
class CaseResult:
    """Aggregated results for one test case across multiple runs."""

    name: str
    complexity: str
    prompt: str
    n_runs: int
    runs: list[RunMetrics] = field(default_factory=list)

    def _values(self, attr: str) -> list[float]:
        return [getattr(r, attr) for r in self.runs]

    def summary(self) -> dict[str, Any]:
        """Compute mean/std/min/max for key metrics."""
        metrics = {}
        for attr in [
            "prompt_tokens",
            "completion_tokens",
            "prefill_ms",
            "decode_ms",
            "inference_ms",
            "latency_ms",
            "prompt_tps",
            "generation_tps",
            "peak_memory_gb",
        ]:
            vals = self._values(attr)
            if not vals:
                continue
            metrics[attr] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
        # Include output text from last run
        if self.runs:
            metrics["output_text"] = self.runs[-1].output_text
        return metrics


@dataclass
class EvalReport:
    """Full evaluation report with comparison support."""

    model: str
    device: str
    timestamp: str
    cases: list[CaseResult] = field(default_factory=list)

    def print(self) -> None:
        """Print human-readable report."""
        print(f"\n{'=' * 70}")
        print(f"  EVAL REPORT — {self.model}")
        print(f"  Device: {self.device}  |  {self.timestamp}")
        print(f"{'=' * 70}")

        for case in self.cases:
            s = case.summary()
            print(f"\n  [{case.name}] complexity={case.complexity}, runs={case.n_runs}")
            print(f"  Prompt: {case.prompt[:60]}...")

            # Key metrics table
            fmt = "    {:<20s} {:>10s} {:>10s}"
            print(fmt.format("Metric", "Mean", "Std"))
            print(f"    {'-' * 42}")

            rows = [
                ("prompt_tokens", "{:.0f}", "{:.0f}"),
                ("prefill_ms", "{:.1f}", "{:.1f}"),
                ("decode_ms", "{:.1f}", "{:.1f}"),
                ("inference_ms", "{:.1f}", "{:.1f}"),
                ("latency_ms", "{:.1f}", "{:.1f}"),
                ("prompt_tps", "{:.0f}", "{:.0f}"),
                ("generation_tps", "{:.1f}", "{:.1f}"),
                ("peak_memory_gb", "{:.2f}", "{:.3f}"),
            ]

            for attr, mfmt, sfmt in rows:
                if attr not in s:
                    continue
                v = s[attr]
                print(
                    fmt.format(
                        attr,
                        mfmt.format(v["mean"]),
                        sfmt.format(v["std"]),
                    )
                )

            if "output_text" in s:
                text = s["output_text"][:80]
                print(f"    Output: {text}...")

        print(f"\n{'=' * 70}\n")

    def save(self, path: str) -> None:
        """Save report as JSON for later comparison."""
        data = {
            "model": self.model,
            "device": self.device,
            "timestamp": self.timestamp,
            "cases": [],
        }
        for case in self.cases:
            data["cases"].append(
                {
                    "name": case.name,
                    "complexity": case.complexity,
                    "prompt": case.prompt,
                    "n_runs": case.n_runs,
                    "summary": case.summary(),
                }
            )
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Report saved to {path}")

    @staticmethod
    def compare(path_a: str, path_b: str) -> None:
        """Compare two saved reports and print delta."""
        a = json.loads(Path(path_a).read_text())
        b = json.loads(Path(path_b).read_text())

        print(f"\n{'=' * 70}")
        print("  COMPARISON")
        print(f"  A: {a['model']} ({a['timestamp']})")
        print(f"  B: {b['model']} ({b['timestamp']})")
        print(f"{'=' * 70}")

        cases_b = {c["name"]: c for c in b["cases"]}

        for ca in a["cases"]:
            cb = cases_b.get(ca["name"])
            if cb is None:
                continue

            print(f"\n  [{ca['name']}]")
            fmt = "    {:<20s} {:>10s} {:>10s} {:>10s}"
            print(fmt.format("Metric", "A", "B", "Delta"))
            print(f"    {'-' * 52}")

            for attr in [
                "prompt_tokens",
                "prefill_ms",
                "decode_ms",
                "inference_ms",
                "latency_ms",
                "peak_memory_gb",
            ]:
                sa = ca["summary"].get(attr, {})
                sb = cb["summary"].get(attr, {})
                if not sa or not sb:
                    continue
                ma, mb = sa["mean"], sb["mean"]
                if ma > 0:
                    delta = ((mb - ma) / ma) * 100
                    sign = "+" if delta > 0 else ""
                    print(
                        fmt.format(
                            attr,
                            f"{ma:.1f}",
                            f"{mb:.1f}",
                            f"{sign}{delta:.1f}%",
                        )
                    )

            # Compare output text
            text_a = ca["summary"].get("output_text", "")
            text_b = cb["summary"].get("output_text", "")
            if text_a and text_b:
                print(f"    A output: {text_a[:70]}...")
                print(f"    B output: {text_b[:70]}...")

        print(f"\n{'=' * 70}\n")


# ── Eval Suite ──────────────────────────────────────────────────────────────


DEFAULT_PROMPT = "Describe what you see in detail."

DEFAULT_CASES = [
    ("solid", "solid", "Minimal visual complexity — uniform color"),
    ("gradient", "gradient", "Low complexity — smooth gradient"),
    ("scene", "scene", "Simulated indoor scene with objects"),
    ("medium", "medium", "Medium complexity — color blocks"),
    ("high", "high", "Maximum complexity — random noise"),
]


class EvalSuite:
    """Run standardized eval benchmarks on a TrioCore engine."""

    def __init__(
        self,
        engine: TrioCore,
        *,
        n_runs: int = 3,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 64,
        height: int = 480,
        width: int = 640,
    ):
        self.engine = engine
        self.n_runs = n_runs
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.height = height
        self.width = width

    def _build_cases(self) -> list[EvalCase]:
        """Build default test cases with synthetic frames."""
        cases = []
        for name, complexity, _desc in DEFAULT_CASES:
            frames = generate_test_frames(
                complexity=complexity,
                height=self.height,
                width=self.width,
                n_frames=2,  # minimum for mlx_vlm
                seed=42,
            )
            cases.append(
                EvalCase(
                    name=name,
                    frames=frames,
                    prompt=self.prompt,
                    complexity=complexity,
                )
            )
        return cases

    def run(self, cases: list[EvalCase] | None = None) -> EvalReport:
        """Run all test cases and collect metrics."""
        if cases is None:
            cases = self._build_cases()

        health = self.engine.health()
        report = EvalReport(
            model=health["model"],
            device=health["backend"]["device"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for case in cases:
            print(f"  Running: {case.name} ({self.n_runs} runs)...", end=" ", flush=True)
            case_result = CaseResult(
                name=case.name,
                complexity=case.complexity,
                prompt=case.prompt,
                n_runs=self.n_runs,
            )

            for i in range(self.n_runs):
                result = self.engine.analyze_video(
                    case.frames,
                    case.prompt,
                    max_tokens=self.max_tokens,
                )
                run = RunMetrics.from_result(result.metrics, result.text)
                case_result.runs.append(run)

            case_result.n_runs = len(case_result.runs)
            report.cases.append(case_result)
            # Show quick inline result
            s = case_result.summary()
            inf = s.get("inference_ms", {}).get("mean", 0)
            pt = s.get("prompt_tokens", {}).get("mean", 0)
            mem = s.get("peak_memory_gb", {}).get("mean", 0)
            print(f"inference={inf:.0f}ms  tokens={pt:.0f}  mem={mem:.2f}GB")

        return report
