"""Standard VLM benchmarks for measuring quality degradation.

Supports:
  - POPE (Object Hallucination) — yes/no questions, accuracy/F1
  - TextVQA — OCR-based QA, accuracy via exact match
  - Custom QA — user-provided image+question+answer sets

Usage:
    from trio_core.eval_benchmarks import POPEBenchmark, BenchmarkRunner
    bench = POPEBenchmark(split="random", max_samples=100)
    runner = BenchmarkRunner(engine)
    result = runner.run(bench)
    result.print()
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class BenchmarkSample:
    """A single benchmark question."""
    id: str
    image: np.ndarray  # (H, W, 3) uint8 or (C, H, W) float32
    question: str
    answer: str  # ground truth
    category: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of running one sample."""
    id: str
    question: str
    answer_gt: str
    answer_pred: str
    correct: bool
    latency_ms: float = 0.0
    prompt_tokens: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    name: str
    n_samples: int
    predictions: list[PredictionResult] = field(default_factory=list)
    total_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.correct for p in self.predictions) / len(self.predictions)

    @property
    def yes_rate(self) -> float:
        """Fraction of predictions that are 'yes' (for POPE bias analysis)."""
        if not self.predictions:
            return 0.0
        yes = sum(1 for p in self.predictions if "yes" in p.answer_pred.lower())
        return yes / len(self.predictions)

    @property
    def f1(self) -> float:
        """Binary F1 for yes/no tasks."""
        tp = fp = fn = tn = 0
        for p in self.predictions:
            pred_yes = "yes" in p.answer_pred.lower()
            gt_yes = "yes" in p.answer_gt.lower()
            if pred_yes and gt_yes:
                tp += 1
            elif pred_yes and not gt_yes:
                fp += 1
            elif not pred_yes and gt_yes:
                fn += 1
            else:
                tn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def avg_latency_ms(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.latency_ms for p in self.predictions) / len(self.predictions)

    @property
    def avg_prompt_tokens(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.prompt_tokens for p in self.predictions) / len(self.predictions)

    def print(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Samples: {self.n_samples}  |  Time: {self.total_time_s:.1f}s")
        print(f"{'='*60}")
        print(f"  Accuracy:         {self.accuracy:.1%}")
        print(f"  F1:               {self.f1:.3f}")
        print(f"  Yes Rate:         {self.yes_rate:.1%}")
        print(f"  Avg Latency:      {self.avg_latency_ms:.0f}ms")
        print(f"  Avg Prompt Tokens:{self.avg_prompt_tokens:.0f}")
        if self.metadata:
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")
        print(f"{'='*60}\n")

    def save(self, path: str) -> None:
        data = {
            "name": self.name,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "yes_rate": self.yes_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "total_time_s": self.total_time_s,
            "metadata": self.metadata,
            "predictions": [
                {
                    "id": p.id,
                    "question": p.question,
                    "answer_gt": p.answer_gt,
                    "answer_pred": p.answer_pred,
                    "correct": p.correct,
                    "latency_ms": p.latency_ms,
                    "prompt_tokens": p.prompt_tokens,
                }
                for p in self.predictions
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Results saved to {path}")

    @staticmethod
    def compare(path_a: str, path_b: str) -> None:
        a = json.loads(Path(path_a).read_text())
        b = json.loads(Path(path_b).read_text())

        print(f"\n{'='*60}")
        print(f"  BENCHMARK COMPARISON")
        print(f"  A: {a['name']}  ({a.get('metadata', {}).get('backend', '?')})")
        print(f"  B: {b['name']}  ({b.get('metadata', {}).get('backend', '?')})")
        print(f"{'='*60}")

        metrics = [
            ("accuracy", "{:.1%}"),
            ("f1", "{:.3f}"),
            ("yes_rate", "{:.1%}"),
            ("avg_latency_ms", "{:.0f}ms"),
            ("avg_prompt_tokens", "{:.0f}"),
        ]

        fmt = "  {:<20s} {:>12s} {:>12s} {:>12s}"
        print(fmt.format("Metric", "A", "B", "Delta"))
        print(f"  {'-'*56}")

        for key, vfmt in metrics:
            va = a.get(key, 0)
            vb = b.get(key, 0)
            if va > 0:
                delta = ((vb - va) / va) * 100
                sign = "+" if delta > 0 else ""
                print(fmt.format(
                    key,
                    vfmt.format(va),
                    vfmt.format(vb),
                    f"{sign}{delta:.1f}%",
                ))

        print(f"{'='*60}\n")


# ── Benchmark Definitions ──────────────────────────────────────────────────


class Benchmark(ABC):
    """Base class for VLM benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self) -> list[BenchmarkSample]:
        """Load benchmark samples."""
        ...

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """Judge if prediction is correct."""
        return self._normalize(prediction) == self._normalize(sample.answer)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for yes/no comparison."""
        text = text.strip().lower()
        # Check for yes/no indicators in the full response
        # "Yes, there is..." → yes
        # "There is no..." / "No, there isn't..." → no
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        # Handle indirect negation: "There is no...", "I don't see..."
        neg_patterns = ["there is no", "there are no", "there isn't", "there aren't",
                        "i don't see", "i do not see", "no,", "not present",
                        "no existence", "does not", "doesn't"]
        for pat in neg_patterns:
            if pat in text:
                return "no"
        # If it contains "yes" anywhere, treat as yes
        if "yes" in text:
            return "yes"
        return text


class POPEBenchmark(Benchmark):
    """POPE: Polling-based Object Probing Evaluation.

    Tests object hallucination with yes/no questions.
    Dataset: lmms-lab/POPE on HuggingFace (9000 samples, COCO images).

    Splits: random, popular, adversarial (3000 each)
    """

    PROMPT_TEMPLATE = "{question} Answer yes or no."

    def __init__(
        self,
        split: str = "random",
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        n = f" (n={self.max_samples})" if self.max_samples else ""
        return f"POPE-{self.split}{n}"

    def load(self) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Install 'datasets' and 'Pillow': pip install datasets Pillow"
            )

        logger.info("Loading POPE dataset (split=%s)...", self.split)

        ds = load_dataset(
            "lmms-lab/POPE",
            "Full",
            split=self.split,
            cache_dir=self.cache_dir,
        )

        samples = []
        n = self.max_samples or len(ds)
        for i in range(min(n, len(ds))):
            item = ds[i]
            # Convert PIL image to numpy (H, W, 3) → (1, 3, H, W) float32
            img = np.array(item["image"])
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            # (H, W, 3) uint8 → (1, 3, H, W) float32
            frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames = frames[np.newaxis]  # (1, 3, H, W)

            samples.append(BenchmarkSample(
                id=str(item.get("question_id", i)),
                image=frames,
                question=item["question"],
                answer=item["answer"].lower(),
                category=item.get("category", self.split),
            ))

        logger.info("Loaded %d POPE samples", len(samples))
        return samples


class TextVQABenchmark(Benchmark):
    """TextVQA: Visual Question Answering requiring text reading.

    Tests OCR capability.
    Dataset: lmms-lab/textvqa on HuggingFace.
    """

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        n = f" (n={self.max_samples})" if self.max_samples else ""
        return f"TextVQA{n}"

    def load(self) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        logger.info("Loading TextVQA dataset...")
        ds = load_dataset(
            "lmms-lab/textvqa",
            split="validation",
            cache_dir=self.cache_dir,
        )

        samples = []
        n = self.max_samples or len(ds)
        for i in range(min(n, len(ds))):
            item = ds[i]
            img = np.array(item["image"])
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames = frames[np.newaxis]

            # TextVQA has a list of answers; use the most common
            answers = item.get("answers", [])
            if isinstance(answers, list) and answers:
                answer = max(set(answers), key=answers.count)
            else:
                answer = str(answers)

            samples.append(BenchmarkSample(
                id=str(item.get("question_id", i)),
                image=frames,
                question=item["question"],
                answer=answer,
                category="textvqa",
            ))

        logger.info("Loaded %d TextVQA samples", len(samples))
        return samples

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """TextVQA uses relaxed matching — answer appears in prediction."""
        pred = prediction.strip().lower()
        gt = sample.answer.strip().lower()
        return gt in pred or pred in gt


class CustomBenchmark(Benchmark):
    """Load a custom benchmark from a JSON file.

    Format:
    [
        {
            "id": "1",
            "image_path": "path/to/image.jpg",
            "question": "What color is the car?",
            "answer": "red"
        },
        ...
    ]
    """

    def __init__(self, path: str, name: str = "custom"):
        self.path = path
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> list[BenchmarkSample]:
        import cv2

        data = json.loads(Path(self.path).read_text())
        samples = []

        for item in data:
            img = cv2.imread(item["image_path"])
            if img is None:
                logger.warning("Could not load image: %s", item["image_path"])
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames = frames[np.newaxis]

            samples.append(BenchmarkSample(
                id=str(item.get("id", len(samples))),
                image=frames,
                question=item["question"],
                answer=item["answer"],
                category=item.get("category", ""),
            ))

        return samples


# ── Benchmark Runner ───────────────────────────────────────────────────────


class BenchmarkRunner:
    """Run benchmarks on a TrioCore engine."""

    def __init__(
        self,
        engine,
        *,
        max_tokens: int = 16,
        prompt_template: str | None = None,
    ):
        self.engine = engine
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template

    def run(self, benchmark: Benchmark) -> BenchmarkResult:
        """Run a benchmark and return results."""
        samples = benchmark.load()

        health = self.engine.health()
        result = BenchmarkResult(
            name=benchmark.name,
            n_samples=len(samples),
            metadata={
                "backend": health["backend"]["backend"],
                "model": health["model"],
                "device": health["backend"]["device"],
                "max_tokens": self.max_tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        t0 = time.time()

        for i, sample in enumerate(samples):
            question = sample.question
            # Use benchmark-specific template, then user override
            if self.prompt_template:
                question = self.prompt_template.format(question=question)
            elif hasattr(benchmark, 'PROMPT_TEMPLATE'):
                question = benchmark.PROMPT_TEMPLATE.format(question=question)

            # Analyze
            tic = time.perf_counter()
            out = self.engine.analyze_video(
                sample.image,
                question,
                max_tokens=self.max_tokens,
            )
            latency = (time.perf_counter() - tic) * 1000

            pred_text = out.text.strip()
            correct = benchmark.judge(sample, pred_text)

            result.predictions.append(PredictionResult(
                id=sample.id,
                question=sample.question,
                answer_gt=sample.answer,
                answer_pred=pred_text,
                correct=correct,
                latency_ms=latency,
                prompt_tokens=out.metrics.prompt_tokens,
            ))

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                running_acc = sum(p.correct for p in result.predictions) / len(result.predictions)
                print(f"  [{i+1}/{len(samples)}] accuracy={running_acc:.1%}", flush=True)

        result.total_time_s = time.time() - t0
        return result
