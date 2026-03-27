"""Standard VLM benchmarks for measuring quality degradation.

Supports:
  - POPE (Object Hallucination) — yes/no questions, accuracy/F1
  - TextVQA — OCR-based QA, accuracy via exact match
  - GQA — Real-world visual reasoning (spatial, attributes, counting)
  - MMBench — Multi-ability benchmark (20 dimensions, multiple choice)
  - MVBench — Video understanding (20 temporal tasks, multiple choice)
  - SurveillanceVQA — Surveillance video QA (detection, classification, description)
  - Custom QA — user-provided image+question+answer sets

Usage:
    from trio_core.eval_benchmarks import POPEBenchmark, MVBenchBenchmark, BenchmarkRunner
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
    # Performance metrics (from InferenceMetrics)
    prompt_tps: float = 0.0  # prefill tokens/sec
    generation_tps: float = 0.0  # decode tokens/sec
    prefill_ms: float = 0.0  # prefill latency
    decode_ms: float = 0.0  # decode latency
    peak_memory_gb: float = 0.0  # peak metal memory
    completion_tokens: int = 0  # output token count
    metadata: dict = field(default_factory=dict)


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
    def recall(self) -> float:
        """True positive rate: TP / (TP + FN) for yes/no tasks."""
        tp = fn = 0
        for p in self.predictions:
            gt_yes = "yes" in p.answer_gt.lower()
            if gt_yes:
                if "yes" in p.answer_pred.lower():
                    tp += 1
                else:
                    fn += 1
        return tp / max(tp + fn, 1)

    @property
    def specificity(self) -> float:
        """True negative rate: TN / (TN + FP) for yes/no tasks."""
        tn = fp = 0
        for p in self.predictions:
            gt_yes = "yes" in p.answer_gt.lower()
            if not gt_yes:
                if "yes" not in p.answer_pred.lower():
                    tn += 1
                else:
                    fp += 1
        return tn / max(tn + fp, 1)

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

    @property
    def avg_prompt_tps(self) -> float:
        vals = [p.prompt_tps for p in self.predictions if p.prompt_tps > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_generation_tps(self) -> float:
        vals = [p.generation_tps for p in self.predictions if p.generation_tps > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_prefill_ms(self) -> float:
        vals = [p.prefill_ms for p in self.predictions if p.prefill_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_decode_ms(self) -> float:
        vals = [p.decode_ms for p in self.predictions if p.decode_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_peak_memory_gb(self) -> float:
        vals = [p.peak_memory_gb for p in self.predictions if p.peak_memory_gb > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def per_category_accuracy(self) -> dict[str, tuple[float, int]]:
        """Per-category accuracy: {category: (accuracy, count)}."""
        from collections import defaultdict

        cats: dict[str, list[bool]] = defaultdict(list)
        for p in self.predictions:
            cat = p.metadata.get("category", "") if p.metadata else ""
            cats[cat].append(p.correct)
        return {cat: (sum(vals) / len(vals), len(vals)) for cat, vals in sorted(cats.items())}

    def print(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {self.name}")
        print(f"  Samples: {self.n_samples}  |  Time: {self.total_time_s:.1f}s")
        print(f"{'=' * 60}")
        print(f"  Accuracy:         {self.accuracy:.1%}")
        print(f"  F1:               {self.f1:.3f}")
        print(f"  Recall:           {self.recall:.1%}")
        print(f"  Specificity:      {self.specificity:.1%}")
        print(f"  Yes Rate:         {self.yes_rate:.1%}")
        print(f"  Avg Latency:      {self.avg_latency_ms:.0f}ms")
        print(f"  Avg Prompt Tokens:{self.avg_prompt_tokens:.0f}")
        if self.avg_prefill_ms > 0:
            print(f"  Avg Prefill:      {self.avg_prefill_ms:.0f}ms")
            print(f"  Avg Decode:       {self.avg_decode_ms:.0f}ms")
            print(f"  Avg Prompt TPS:   {self.avg_prompt_tps:.0f}")
            print(f"  Avg Gen TPS:      {self.avg_generation_tps:.0f}")
            print(f"  Peak Memory:      {self.avg_peak_memory_gb:.2f}GB")
        if self.metadata:
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")
        # Per-category breakdown (for MVBench tasks, etc.)
        cat_acc = self.per_category_accuracy
        if len(cat_acc) > 1:
            print("\n  Per-category accuracy:")
            for cat, (acc, cnt) in cat_acc.items():
                if cat:
                    print(f"    {cat:<30s} {acc:.1%}  (n={cnt})")
        print(f"{'=' * 60}\n")

    def save(self, path: str) -> None:
        data = {
            "name": self.name,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "recall": self.recall,
            "specificity": self.specificity,
            "yes_rate": self.yes_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_prefill_ms": self.avg_prefill_ms,
            "avg_decode_ms": self.avg_decode_ms,
            "avg_prompt_tps": self.avg_prompt_tps,
            "avg_generation_tps": self.avg_generation_tps,
            "avg_peak_memory_gb": self.avg_peak_memory_gb,
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
                    "prompt_tps": p.prompt_tps,
                    "generation_tps": p.generation_tps,
                    "prefill_ms": p.prefill_ms,
                    "decode_ms": p.decode_ms,
                    "peak_memory_gb": p.peak_memory_gb,
                    "completion_tokens": p.completion_tokens,
                    "category": p.metadata.get("category", ""),
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

        print(f"\n{'=' * 60}")
        print("  BENCHMARK COMPARISON")
        print(f"  A: {a['name']}  ({a.get('metadata', {}).get('backend', '?')})")
        print(f"  B: {b['name']}  ({b.get('metadata', {}).get('backend', '?')})")
        print(f"{'=' * 60}")

        metrics = [
            ("accuracy", "{:.1%}"),
            ("f1", "{:.3f}"),
            ("yes_rate", "{:.1%}"),
            ("avg_latency_ms", "{:.0f}ms"),
            ("avg_prompt_tokens", "{:.0f}"),
        ]

        fmt = "  {:<20s} {:>12s} {:>12s} {:>12s}"
        print(fmt.format("Metric", "A", "B", "Delta"))
        print(f"  {'-' * 56}")

        for key, vfmt in metrics:
            va = a.get(key, 0)
            vb = b.get(key, 0)
            if va > 0:
                delta = ((vb - va) / va) * 100
                sign = "+" if delta > 0 else ""
                print(
                    fmt.format(
                        key,
                        vfmt.format(va),
                        vfmt.format(vb),
                        f"{sign}{delta:.1f}%",
                    )
                )

        print(f"{'=' * 60}\n")


# ── Benchmark Definitions ──────────────────────────────────────────────────


class Benchmark(ABC):
    """Base class for VLM benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str: ...

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
        import re

        # Strip <think>...</think> blocks (Qwen3.5 thinking mode)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Take only first line to avoid hallucinated continuations
        text = text.split("\n")[0].strip().lower()
        # Check for yes/no indicators in the full response
        # "Yes, there is..." → yes
        # "There is no..." / "No, there isn't..." → no
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        # Handle indirect negation: "There is no...", "I don't see..."
        neg_patterns = [
            "there is no",
            "there are no",
            "there isn't",
            "there aren't",
            "i don't see",
            "i do not see",
            "no,",
            "not present",
            "no existence",
            "does not",
            "doesn't",
            "cannot see",
            "can't see",
            "not visible",
            "is not",
            "are not",
            "without any",
            "no sign of",
            "absent",
        ]
        for pat in neg_patterns:
            if pat in text:
                return "no"
        # If it contains "yes" anywhere, treat as yes
        if "yes" in text:
            return "yes"
        # Handle affirmative descriptions: "The image shows...", "There is a..."
        aff_patterns = [
            "there is a",
            "there are",
            "the image shows",
            "the image features",
            "the image depicts",
            "can be seen",
            "is visible",
            "are visible",
            "is present",
            "are present",
            "i can see",
            "shows a",
            "depicts a",
            "features a",
            "contains a",
            "includes a",
        ]
        for pat in aff_patterns:
            if pat in text:
                return "yes"
        return text


class POPEBenchmark(Benchmark):
    """POPE: Polling-based Object Probing Evaluation.

    Tests object hallucination with yes/no questions.
    Dataset: lmms-lab/POPE on HuggingFace (9000 samples, COCO images).

    Splits: random, popular, adversarial (3000 each)
    """

    PROMPT_TEMPLATE = "{question} Answer with only one word: yes or no."

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
            raise ImportError("Install 'datasets' and 'Pillow': pip install datasets Pillow")

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

            samples.append(
                BenchmarkSample(
                    id=str(item.get("question_id", i)),
                    image=frames,
                    question=item["question"],
                    answer=item["answer"].lower(),
                    category=item.get("category", self.split),
                )
            )

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

            samples.append(
                BenchmarkSample(
                    id=str(item.get("question_id", i)),
                    image=frames,
                    question=item["question"],
                    answer=answer,
                    category="textvqa",
                )
            )

        logger.info("Loaded %d TextVQA samples", len(samples))
        return samples

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """TextVQA uses relaxed matching — answer appears in prediction."""
        pred = prediction.strip().lower()
        gt = sample.answer.strip().lower()
        return gt in pred or pred in gt


class GQABenchmark(Benchmark):
    """GQA: Real-World Visual Reasoning and Compositional QA.

    Tests compositional visual reasoning on real-world images (COCO).
    Dataset: lmms-lab/GQA on HuggingFace.
    Uses testdev_balanced split (12,578 questions).

    Questions test spatial relations, attributes, counting, comparisons
    on real-world scenes — the most relevant benchmark for camera/webcam
    use cases.
    """

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        n = f" (n={self.max_samples})" if self.max_samples else ""
        return f"GQA{n}"

    def load(self) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        logger.info("Loading GQA dataset (testdev_balanced)...")

        # GQA has separate image and instruction datasets
        ds_instructions = load_dataset(
            "lmms-lab/GQA",
            "testdev_balanced_instructions",
            split="testdev",
            cache_dir=self.cache_dir,
        )
        ds_images = load_dataset(
            "lmms-lab/GQA",
            "testdev_balanced_images",
            split="testdev",
            cache_dir=self.cache_dir,
        )

        # Build image lookup by id
        logger.info("Building image index (%d images)...", len(ds_images))
        image_lookup: dict[str, Any] = {}
        for img_item in ds_images:
            image_lookup[img_item["id"]] = img_item["image"]

        samples = []
        n = self.max_samples or len(ds_instructions)
        for i in range(min(n, len(ds_instructions))):
            item = ds_instructions[i]
            image_id = item["imageId"]

            if image_id not in image_lookup:
                logger.warning("Image %s not found, skipping", image_id)
                continue

            pil_img = image_lookup[image_id]
            img = np.array(pil_img)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[2] == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames = frames[np.newaxis]

            samples.append(
                BenchmarkSample(
                    id=str(item["id"]),
                    image=frames,
                    question=item["question"],
                    answer=item["answer"].lower(),
                    category=item.get("types", {}).get("structural", ""),
                )
            )

        logger.info("Loaded %d GQA samples", len(samples))
        return samples

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """GQA uses relaxed matching — short answers, case-insensitive."""
        pred = prediction.strip().lower()
        gt = sample.answer.strip().lower()
        # Exact match
        if pred == gt:
            return True
        # GT appears at start of prediction (e.g., gt="yes", pred="yes, it is")
        if pred.startswith(gt):
            return True
        # GT appears as a word in prediction
        if re.search(r"\b" + re.escape(gt) + r"\b", pred):
            return True
        return False


class MMBenchBenchmark(Benchmark):
    """MMBench: Multi-ability Multi-modal Benchmark.

    Tests 20 ability dimensions including spatial reasoning, attribute
    recognition, object localization, etc. Multiple-choice format.
    Dataset: lmms-lab/MMBench on HuggingFace (English dev split).
    """

    OPTION_LETTERS = ["A", "B", "C", "D", "E"]

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        n = f" (n={self.max_samples})" if self.max_samples else ""
        return f"MMBench{n}"

    def load(self) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        logger.info("Loading MMBench dataset (en/dev)...")
        ds = load_dataset(
            "lmms-lab/MMBench",
            "en",
            split="dev",
            cache_dir=self.cache_dir,
        )

        samples = []
        n = self.max_samples or len(ds)
        for i in range(min(n, len(ds))):
            item = ds[i]

            img = np.array(item["image"])
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames = frames[np.newaxis]

            # Build multiple-choice question
            options = []
            for letter in self.OPTION_LETTERS:
                opt = item.get(letter)
                if opt and str(opt).strip() and str(opt).strip().lower() != "nan":
                    options.append(f"{letter}. {opt}")
            options_text = "\n".join(options)

            hint = item.get("hint", "")
            hint_text = f"Hint: {hint}\n" if hint and str(hint).strip().lower() != "nan" else ""

            question = f"{hint_text}{item['question']}\n{options_text}\nAnswer with the option letter only."
            answer = item["answer"]

            samples.append(
                BenchmarkSample(
                    id=str(item.get("index", i)),
                    image=frames,
                    question=question,
                    answer=answer,
                    category=item.get("category", ""),
                    metadata={"l2_category": item.get("l2-category", "")},
                )
            )

        logger.info("Loaded %d MMBench samples", len(samples))
        return samples

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """MMBench: match option letter."""
        pred = prediction.strip().upper()
        gt = sample.answer.strip().upper()

        # Direct letter match
        if pred == gt:
            return True
        # First character is the answer letter
        if pred and pred[0] == gt:
            return True
        # Look for pattern like "A." or "(A)" in prediction
        match = re.match(r"^([A-E])[.\s)\]]", pred)
        if match and match.group(1) == gt:
            return True
        return False


class MVBenchBenchmark(Benchmark):
    """MVBench: Multi-modal Video Understanding Benchmark.

    Tests 20 temporal reasoning tasks on video clips. Multiple-choice format.
    Dataset: OpenGVLab/MVBench on HuggingFace.

    Videos must be downloaded separately (17.3GB total). Use download_videos()
    or set video_dir to a pre-downloaded directory.

    Surveillance-relevant tasks: action_sequence, object_interaction,
    state_change, moving_direction, object_existence.
    """

    # Task name → (json_file, video_subfolder, data_type, has_temporal_bounds)
    TASK_CONFIG = {
        "action_sequence": ("action_sequence.json", "star/Charades_v1_480", "video", True),
        "action_prediction": ("action_prediction.json", "star/Charades_v1_480", "video", True),
        "action_antonym": ("action_antonym.json", "ssv2_video", "video", False),
        "fine_grained_action": (
            "fine_grained_action.json",
            "Moments_in_Time_Raw/videos",
            "video",
            False,
        ),
        "unexpected_action": ("unexpected_action.json", "FunQA_test/test", "video", False),
        "object_existence": ("object_existence.json", "clevrer/video_validation", "video", False),
        "object_interaction": ("object_interaction.json", "star/Charades_v1_480", "video", True),
        "object_shuffle": ("object_shuffle.json", "perception/videos", "video", False),
        "moving_direction": ("moving_direction.json", "clevrer/video_validation", "video", False),
        "action_localization": ("action_localization.json", "sta/sta_video", "video", True),
        "scene_transition": ("scene_transition.json", "scene_qa/video", "video", False),
        "action_count": ("action_count.json", "perception/videos", "video", False),
        "moving_count": ("moving_count.json", "clevrer/video_validation", "video", False),
        "moving_attribute": ("moving_attribute.json", "clevrer/video_validation", "video", False),
        "state_change": ("state_change.json", "perception/videos", "video", False),
        "fine_grained_pose": ("fine_grained_pose.json", "nturgbd", "video", False),
        "character_order": ("character_order.json", "perception/videos", "video", False),
        "egocentric_navigation": ("egocentric_navigation.json", "vlnqa", "video", False),
        "episodic_reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),
        "counterfactual_inference": (
            "counterfactual_inference.json",
            "clevrer/video_validation",
            "video",
            False,
        ),
    }

    # Tasks grouped by video source for selective download
    VIDEO_SOURCES = {
        "clevrer": [
            "object_existence",
            "moving_direction",
            "moving_count",
            "moving_attribute",
            "counterfactual_inference",
        ],
        "star": ["action_sequence", "action_prediction", "object_interaction"],
        "perception": ["object_shuffle", "action_count", "state_change", "character_order"],
        "scene_qa": ["scene_transition"],
        "sta": ["action_localization"],
        "ssv2_video": ["action_antonym"],
        "Moments_in_Time_Raw": ["fine_grained_action"],
        "FunQA_test": ["unexpected_action"],
        "vlnqa": ["egocentric_navigation"],
        "tvqa": ["episodic_reasoning"],
        "nturgbd": ["fine_grained_pose"],
    }

    _PROMPT_TEMPLATE = (
        "Watch the video carefully and answer the following question.\n"
        "{question}\n{options}\n"
        "Answer with the option letter only."
    )

    def __init__(
        self,
        video_dir: str,
        tasks: list[str] | None = None,
        max_samples_per_task: int | None = None,
        max_frames: int = 16,
        fps: float = 1.0,
        cache_dir: str | None = None,
    ):
        """
        Args:
            video_dir: Root directory containing extracted video folders
                       (e.g., video_dir/star/Charades_v1_480/ZS9XR.mp4).
            tasks: List of task names to evaluate. None = all available.
            max_samples_per_task: Limit samples per task (for quick testing).
            max_frames: Max frames to extract per video clip.
            fps: Frame extraction rate.
        """
        self.video_dir = Path(video_dir)
        self.tasks = tasks
        self.max_samples_per_task = max_samples_per_task
        self.max_frames = max_frames
        self.fps = fps
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        n_tasks = len(self.tasks) if self.tasks else 20
        n = f" (n={self.max_samples_per_task}/task)" if self.max_samples_per_task else ""
        return f"MVBench-{n_tasks}tasks{n}"

    def load(self) -> list[BenchmarkSample]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("Install 'huggingface_hub': pip install huggingface_hub")

        tasks_to_run = self.tasks or list(self.TASK_CONFIG.keys())
        # Filter to tasks with valid config
        tasks_to_run = [t for t in tasks_to_run if t in self.TASK_CONFIG]

        all_samples = []
        skipped = 0

        for task_name in tasks_to_run:
            json_file, video_subdir, data_type, has_bounds = self.TASK_CONFIG[task_name]

            # Download annotation JSON from HuggingFace
            json_path = hf_hub_download(
                "OpenGVLab/MVBench",
                f"json/{json_file}",
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )
            with open(json_path) as f:
                annotations = json.load(f)

            n = self.max_samples_per_task or len(annotations)
            task_samples = 0

            for item in annotations[:n]:
                video_name = item["video"]
                video_path = self.video_dir / video_subdir / video_name

                # Try with .mp4 extension if not found
                if not video_path.exists() and not video_name.endswith(
                    (".mp4", ".avi", ".webm", ".mkv")
                ):
                    video_path = self.video_dir / video_subdir / f"{video_name}.mp4"

                if not video_path.exists():
                    skipped += 1
                    continue

                # Extract frames from video
                start = item.get("start")
                end = item.get("end")
                if data_type == "frame":
                    frames = self._load_frame_sequence(video_path, start, end)
                else:
                    frames = self._extract_video_frames(str(video_path), start, end)

                if frames is None:
                    skipped += 1
                    continue

                # Build multiple-choice question
                candidates = item["candidates"]
                option_letters = [chr(ord("A") + i) for i in range(len(candidates))]
                options_text = "\n".join(
                    f"{letter}. {cand}" for letter, cand in zip(option_letters, candidates)
                )
                question = self._PROMPT_TEMPLATE.format(
                    question=item["question"], options=options_text
                )

                # Find correct answer letter
                answer_text = item["answer"]
                answer_letter = "A"  # default
                for i, cand in enumerate(candidates):
                    if cand == answer_text:
                        answer_letter = chr(ord("A") + i)
                        break

                all_samples.append(
                    BenchmarkSample(
                        id=f"{task_name}_{task_samples}",
                        image=frames,
                        question=question,
                        answer=answer_letter,
                        category=task_name,
                        metadata={
                            "answer_text": answer_text,
                            "video": video_name,
                        },
                    )
                )
                task_samples += 1

            logger.info("Loaded %d samples for task '%s'", task_samples, task_name)

        if skipped:
            logger.warning(
                "Skipped %d samples (videos not found in %s). "
                "Download videos with MVBenchBenchmark.download_videos().",
                skipped,
                self.video_dir,
            )

        logger.info("Loaded %d total MVBench samples", len(all_samples))
        return all_samples

    def _extract_video_frames(
        self, video_path: str, start: float | None, end: float | None
    ) -> np.ndarray | None:
        """Extract frames from a video file, returning (T, C, H, W) float32."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Determine frame range
        if start is not None and end is not None:
            start_frame = int(start * video_fps)
            end_frame = int(end * video_fps)
        else:
            start_frame = 0
            end_frame = total_frames

        end_frame = min(end_frame, total_frames)
        duration_frames = max(end_frame - start_frame, 1)

        # Sample frames uniformly
        n_frames = min(self.max_frames, duration_frames)
        if n_frames <= 0:
            cap.release()
            return None

        indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to reasonable size for VLM
                h, w = frame.shape[:2]
                if max(h, w) > 480:
                    scale = 480.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                frames.append(frame)

        cap.release()

        if not frames:
            return None

        # Stack to (T, C, H, W) float32
        # All frames should be same size (from same video)
        frames_arr = np.stack(frames)  # (T, H, W, 3)
        frames_arr = frames_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return frames_arr

    def _load_frame_sequence(
        self, frame_dir: Path, start: float | None, end: float | None
    ) -> np.ndarray | None:
        """Load frames from a directory of images (for episodic_reasoning)."""
        import cv2

        if not frame_dir.is_dir():
            # Try as video file
            return self._extract_video_frames(str(frame_dir), start, end)

        frame_files = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if not frame_files:
            return None

        # Subsample
        n_frames = min(self.max_frames, len(frame_files))
        indices = np.linspace(0, len(frame_files) - 1, n_frames, dtype=int)

        frames = []
        for idx in indices:
            img = cv2.imread(str(frame_files[idx]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if max(h, w) > 480:
                    scale = 480.0 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                frames.append(img)

        if not frames:
            return None

        frames_arr = np.stack(frames)
        frames_arr = frames_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return frames_arr

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """Match option letter from prediction."""
        pred = prediction.strip().upper()
        gt = sample.answer.strip().upper()

        # Direct letter match
        if pred == gt:
            return True
        # First character is the answer letter
        if pred and pred[0] == gt:
            return True
        # Pattern like "A." or "(A)"
        match = re.match(r"^([A-Z])[.\s)\]]", pred)
        if match and match.group(1) == gt:
            return True
        # Answer text match as fallback
        answer_text = sample.metadata.get("answer_text", "")
        if answer_text and answer_text.lower() in prediction.lower():
            return True
        return False

    @staticmethod
    def download_videos(
        output_dir: str,
        sources: list[str] | None = None,
    ) -> None:
        """Download MVBench video zip files from HuggingFace and extract them.

        Args:
            output_dir: Directory to extract videos into.
            sources: Which video sources to download (e.g., ["clevrer", "star"]).
                     None = all sources. Total ~17.3GB.
        """
        from huggingface_hub import hf_hub_download

        # Zip file mapping
        source_zips = {
            "clevrer": "video/clevrer.zip",
            "star": "video/star.zip",
            "perception": "video/perception.zip",
            "scene_qa": "video/scene_qa.zip",
            "sta": "video/sta.zip",
            "ssv2_video": "video/ssv2_video.zip",
            "Moments_in_Time_Raw": "video/Moments_in_Time_Raw.zip",
            "FunQA_test": "video/FunQA_test.zip",
            "vlnqa": "video/vlnqa.zip",
            "tvqa": "video/tvqa.zip",
            # nturgbd needs manual download from ROSE Lab
        }

        sources_to_dl = sources or list(source_zips.keys())
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import zipfile

        for source in sources_to_dl:
            if source not in source_zips:
                logger.warning("Unknown source '%s', skipping", source)
                continue

            zip_path_hf = source_zips[source]
            logger.info("Downloading %s...", zip_path_hf)
            local_zip = hf_hub_download(
                "OpenGVLab/MVBench",
                zip_path_hf,
                repo_type="dataset",
            )

            logger.info("Extracting %s to %s...", source, out)
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(out)

            logger.info("Done: %s", source)

    @staticmethod
    def available_tasks() -> list[str]:
        """Return list of all 20 MVBench task names."""
        return list(MVBenchBenchmark.TASK_CONFIG.keys())

    @staticmethod
    def tasks_for_source(source: str) -> list[str]:
        """Return task names that use a given video source."""
        return MVBenchBenchmark.VIDEO_SOURCES.get(source, [])


class SurveillanceVQABenchmark(Benchmark):
    """SurveillanceVQA-589K: Surveillance video question answering benchmark.

    Based on: https://arxiv.org/abs/2505.12589
    Dataset: https://huggingface.co/datasets/fei213/SurveillanceVQA-589K

    Tests VLM performance on surveillance/monitoring scenarios using QA pairs
    from 4 video sources (UCF-Crime, MSAD, MEVA, NWPU Campus).

    Question types:
      Abnormal: detection (yes/no), classification, subject, description, cause, result
      Normal: summary, generic, temporal, short_temporal, spatial, reasoning

    Requires:
      - QA annotations: download from HuggingFace (test_datasets.zip)
      - Source videos: UCF-Crime videos in video_dir/
      - Category metadata: UCF_qwen_category.json for clip timestamps
    """

    # QA types for abnormal clips
    ABNORMAL_QA_TYPES = [
        "detection_qa_pairs",
        "classification_qa_pairs",
        "subject_qa_pairs",
        "description_qa_pairs",
        "cause_qa_pairs",
        "result_qa_pairs",
    ]
    # QA types for normal clips
    NORMAL_QA_TYPES = [
        "summary_qa_pairs",
        "generic_qa_pairs",
        "temporal_qa_pairs",
        "spatial_qa_pairs",
        "reasoning_qa_pairs",
        "short_temporal_qa_pairs",
    ]

    # Per-category prompt suffixes to constrain model output format
    _PROMPT_SUFFIX = {
        "detection": " Answer with only yes or no.",
        "classification": " Answer with the anomaly type in a few words.",
        "subject": " Answer briefly.",
        "description": " Describe briefly.",
        "cause": " Answer briefly.",
        "result": " Answer briefly.",
        "summary": " Summarize briefly.",
        "generic": "",
        "temporal": " Answer briefly.",
        "short_temporal": " Answer briefly.",
        "spatial": " Answer briefly.",
        "reasoning": " Answer briefly.",
    }

    def __init__(
        self,
        data_dir: str,
        video_dir: str | None = None,
        qa_type: str = "detection",
        sources: list[str] | None = None,
        max_samples: int | None = None,
        max_frames: int = 8,
    ):
        """
        Args:
            data_dir: Directory containing test_datasets/ and category JSONs.
            video_dir: Directory containing source video files. If None, uses
                       data_dir/videos/.
            qa_type: Which QA type to evaluate. Options:
                     "detection" — yes/no anomaly detection (fastest, POPE-like)
                     "classification" — anomaly type classification
                     "description" — scene description (open-ended)
                     "all_abnormal" — all 6 abnormal QA types
                     "all" — all 12 QA types (abnormal + normal)
            sources: Video sources to include. Default: ["UCF"].
                     Options: "UCF", "MSAD", "MEVA", "NWPU_Test", "NWPU_Train"
            max_samples: Max total samples to load.
            max_frames: Max frames to extract per video clip.
        """
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir) if video_dir else self.data_dir / "videos"
        self.qa_type = qa_type
        self.sources = sources or ["UCF"]
        self.max_samples = max_samples
        self.max_frames = max_frames

    @property
    def name(self) -> str:
        n = f" (n={self.max_samples})" if self.max_samples else ""
        src = "+".join(self.sources)
        return f"SurveillanceVQA-{self.qa_type}-{src}{n}"

    def _get_qa_type_keys(self) -> list[str]:
        """Return list of QA type keys to extract based on self.qa_type."""
        if self.qa_type == "detection":
            return ["detection_qa_pairs"]
        elif self.qa_type == "classification":
            return ["classification_qa_pairs"]
        elif self.qa_type == "description":
            return ["description_qa_pairs"]
        elif self.qa_type == "all_abnormal":
            return self.ABNORMAL_QA_TYPES
        elif self.qa_type == "all":
            return self.ABNORMAL_QA_TYPES + self.NORMAL_QA_TYPES
        else:
            return [f"{self.qa_type}_qa_pairs"]

    def _load_category_metadata(self, source: str) -> dict:
        """Load timestamp/caption metadata for video clips."""
        # Try multiple possible locations for category JSON
        candidates = [
            self.data_dir / f"{source}_qwen_category.json",
            self.data_dir / "output" / f"{source}_qwen_category.json",
            self.data_dir
            / "github"
            / "6_find_normal_abnormal"
            / "output"
            / f"{source}_qwen_category.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        logger.warning("Category metadata not found for %s (tried %s)", source, candidates)
        return {}

    def _find_video_file(self, video_name: str) -> Path | None:
        """Find the video file for a given video name."""
        # UCF-Crime videos are organized as Category/VideoName.mp4
        # e.g., Abuse/Abuse002_x264.mp4
        # Extract category from video name (e.g., "Abuse" from "Abuse002_x264")
        category = re.match(r"([A-Za-z_]+?)(\d|_Videos)", video_name)
        if category:
            cat_name = category.group(1).rstrip("_")
        else:
            cat_name = ""

        candidates = [
            self.video_dir / f"{video_name}.mp4",
            self.video_dir / cat_name / f"{video_name}.mp4",
            self.video_dir / "Anomaly-Videos-Part-1" / cat_name / f"{video_name}.mp4",
            self.video_dir / "Anomaly-Videos-Part-2" / cat_name / f"{video_name}.mp4",
            self.video_dir / "Anomaly-Videos-Part-3" / cat_name / f"{video_name}.mp4",
            self.video_dir / "Anomaly-Videos-Part-4" / cat_name / f"{video_name}.mp4",
        ]
        # Also try Normal_Videos subfolders
        if "Normal" in video_name:
            candidates.extend(
                [
                    self.video_dir / "Normal_Videos_of_Events" / f"{video_name}.mp4",
                    self.video_dir / "Testing_Normal_Videos" / f"{video_name}.mp4",
                    self.video_dir / "Testing_Normal_Videos_Anomaly" / f"{video_name}.mp4",
                ]
            )

        for path in candidates:
            if path.exists():
                return path
        return None

    def _extract_clip_frames(
        self, video_path: Path, start: float | None, end: float | None
    ) -> np.ndarray | None:
        """Extract frames from a video clip, returning (T, C, H, W) float32."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if start is not None and end is not None:
            start_frame = int(start * fps)
            end_frame = int(end * fps)
        else:
            start_frame = 0
            end_frame = total_frames

        end_frame = min(end_frame, total_frames)
        duration = max(end_frame - start_frame, 1)
        n_frames = min(self.max_frames, duration)

        if n_frames <= 0:
            cap.release()
            return None

        indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if not frames:
            return None

        # Convert to (T, C, H, W) float32
        arr = np.stack(frames)  # (T, H, W, C)
        arr = arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return arr

    # Standard detection question for synthesizing "No" samples from normal clips
    _DETECTION_QUESTION = "Does this video contain any potentially violent or criminal activities?"

    def load(self) -> list[BenchmarkSample]:
        qa_type_keys = self._get_qa_type_keys()
        # Determine which subdirectories to scan
        include_abnormal = any(k in self.ABNORMAL_QA_TYPES for k in qa_type_keys)
        include_normal = any(k in self.NORMAL_QA_TYPES for k in qa_type_keys)

        # For detection mode: also include normal clips with "No" answers
        # to create a balanced yes/no benchmark (like POPE)
        add_normal_detection = self.qa_type == "detection"

        all_samples = []
        skipped_no_video = 0
        skipped_no_frames = 0

        for source in self.sources:
            # Load category metadata for timestamps
            cat_meta = self._load_category_metadata(source)

            # Scan test subdirectories (abnormal first for balanced sampling)
            subdirs = []
            if include_abnormal:
                subdirs.append(f"{source}_abnormal")
            if include_normal or add_normal_detection:
                subdirs.append(f"{source}_normal")

            # For detection mode with max_samples: reserve half for each class
            normal_budget = None
            if add_normal_detection and self.max_samples:
                abnormal_budget = self.max_samples // 2
                normal_budget = self.max_samples - abnormal_budget

            for subdir_name in subdirs:
                subdir = self.data_dir / "test_datasets" / subdir_name
                if not subdir.exists():
                    logger.warning("Directory not found: %s", subdir)
                    continue

                is_normal_dir = subdir_name.endswith("_normal")
                # Apply per-class budget for balanced detection
                subdir_count = 0
                if add_normal_detection and self.max_samples:
                    budget = normal_budget if is_normal_dir else abnormal_budget
                else:
                    budget = self.max_samples

                json_files = sorted(subdir.glob("*.json"))
                for json_file in json_files:
                    if budget and subdir_count >= budget:
                        break
                    if self.max_samples and len(all_samples) >= self.max_samples:
                        break

                    with open(json_file) as f:
                        qa_data = json.load(f)

                    # Parse filename: VideoName_clipIndex.json
                    stem = json_file.stem  # e.g., "Abuse002_x264_18"
                    parts = stem.rsplit("_", 1)
                    video_name = parts[0]  # "Abuse002_x264"
                    clip_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

                    # Get clip timestamps from category metadata
                    start_time, end_time = None, None
                    if video_name in cat_meta:
                        timestamps = cat_meta[video_name].get("timestamps", [])
                        if clip_idx < len(timestamps):
                            start_time, end_time = timestamps[clip_idx]

                    # Find and load video
                    video_path = self._find_video_file(video_name)
                    if video_path is None:
                        skipped_no_video += 1
                        continue

                    frames = self._extract_clip_frames(video_path, start_time, end_time)
                    if frames is None:
                        skipped_no_frames += 1
                        continue

                    # For normal clips in detection mode: synthesize "No" answer
                    if is_normal_dir and add_normal_detection:
                        q = self._DETECTION_QUESTION + self._PROMPT_SUFFIX.get("detection", "")
                        all_samples.append(
                            BenchmarkSample(
                                id=f"{source}_{stem}_detection_0",
                                image=frames,
                                question=q,
                                answer="No",
                                category="detection",
                                metadata={
                                    "source": source,
                                    "video": video_name,
                                    "clip_idx": clip_idx,
                                    "anomaly_type": "normal",
                                    "subdir": subdir_name,
                                },
                            )
                        )
                        subdir_count += 1
                        continue  # skip QA extraction for normal clips in detection mode

                    # Extract QA pairs of requested types
                    for qa_key in qa_type_keys:
                        if qa_key not in qa_data:
                            continue
                        for qi, qa_pair in enumerate(qa_data[qa_key]):
                            if self.max_samples and len(all_samples) >= self.max_samples:
                                break
                            # Determine category label
                            qa_category = qa_key.replace("_qa_pairs", "")
                            anomaly_type = ""
                            if "classification" in qa_key:
                                anomaly_type = qa_pair["A"]
                            elif "detection" in qa_key:
                                anomaly_type = (
                                    "abnormal" if qa_pair["A"].lower() == "yes" else "normal"
                                )

                            # Apply per-category prompt suffix
                            q = qa_pair["Q"] + self._PROMPT_SUFFIX.get(qa_category, "")

                            all_samples.append(
                                BenchmarkSample(
                                    id=f"{source}_{stem}_{qa_category}_{qi}",
                                    image=frames,
                                    question=q,
                                    answer=qa_pair["A"],
                                    category=qa_category,
                                    metadata={
                                        "source": source,
                                        "video": video_name,
                                        "clip_idx": clip_idx,
                                        "anomaly_type": anomaly_type,
                                        "subdir": subdir_name,
                                    },
                                )
                            )
                    subdir_count += 1

                if self.max_samples and len(all_samples) >= self.max_samples:
                    break

        if skipped_no_video:
            logger.warning(
                "Skipped %d clips (video not found in %s). Download UCF-Crime videos to video_dir.",
                skipped_no_video,
                self.video_dir,
            )
        if skipped_no_frames:
            logger.warning("Skipped %d clips (frame extraction failed).", skipped_no_frames)

        logger.info("Loaded %d SurveillanceVQA samples", len(all_samples))
        return all_samples

    def judge(self, sample: BenchmarkSample, prediction: str) -> bool:
        """Judge prediction against ground truth."""
        category = sample.category
        if category == "detection":
            # Yes/No matching (same as POPE)
            return self._normalize(prediction) == self._normalize(sample.answer)
        elif category == "classification":
            # Check if predicted anomaly type is in the ground truth
            # GT can be "Traffic Accident, People Falling" (multiple types)
            pred_norm = self._normalize(prediction).lower()
            gt_types = [t.strip().lower() for t in sample.answer.split(",")]
            return any(t in pred_norm for t in gt_types)
        else:
            # Open-ended: basic containment check for short answers,
            # or first-word match. Full eval would use LLM judge.
            pred_norm = self._normalize(prediction).lower()
            gt_norm = sample.answer.lower().strip()
            # For very short answers, do exact match
            if len(gt_norm.split()) <= 3:
                return pred_norm == gt_norm or gt_norm in pred_norm
            # For longer answers, check key phrase overlap
            gt_words = set(gt_norm.split())
            pred_words = set(pred_norm.split())
            if not gt_words:
                return False
            overlap = len(gt_words & pred_words) / len(gt_words)
            return overlap > 0.3


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

            samples.append(
                BenchmarkSample(
                    id=str(item.get("id", len(samples))),
                    image=frames,
                    question=item["question"],
                    answer=item["answer"],
                    category=item.get("category", ""),
                )
            )

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
            elif hasattr(benchmark, "PROMPT_TEMPLATE"):
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
            # Strip <think>...</think> blocks (Qwen3.5 thinking mode)
            # Handle both complete and truncated think tags
            pred_text = re.sub(r"<think>.*?</think>", "", pred_text, flags=re.DOTALL).strip()
            if pred_text.startswith("<think>"):
                pred_text = ""  # entirely thinking, no answer produced
            correct = benchmark.judge(sample, pred_text)

            result.predictions.append(
                PredictionResult(
                    id=sample.id,
                    question=sample.question,
                    answer_gt=sample.answer,
                    answer_pred=pred_text,
                    correct=correct,
                    latency_ms=latency,
                    prompt_tokens=out.metrics.prompt_tokens,
                    prompt_tps=out.metrics.prompt_tps,
                    generation_tps=out.metrics.tokens_per_sec,
                    prefill_ms=out.metrics.prefill_ms,
                    decode_ms=out.metrics.decode_ms,
                    peak_memory_gb=out.metrics.peak_memory_gb,
                    completion_tokens=out.metrics.completion_tokens,
                    metadata={"category": sample.category},
                )
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                running_acc = sum(p.correct for p in result.predictions) / len(result.predictions)
                print(f"  [{i + 1}/{len(samples)}] accuracy={running_acc:.1%}", flush=True)

        result.total_time_s = time.time() - t0
        return result
