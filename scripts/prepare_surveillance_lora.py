#!/usr/bin/env python3
"""Prepare SurveillanceVQA data for mlx-vlm LoRA training.

Converts UCF-Crime video clips + QA annotations into HuggingFace datasets
format for mlx-vlm LoRA fine-tuning.

Usage:
    source .venv-lora/bin/activate
    python scripts/prepare_surveillance_lora.py --max-samples 100  # smoke test
    python scripts/prepare_surveillance_lora.py                     # full dataset
"""

import argparse
import json
import random
import re
from pathlib import Path

import cv2
from PIL import Image

# --- Video file discovery ---

# UCF-Crime video dirs across parts
ANOMALY_PARTS = [
    "Anomaly-Videos-Part-1",
    "Anomaly-Videos-Part-2",
    "Anomaly-Videos-Part-3",
    "Anomaly-Videos-Part-4",
]

# Crime categories → which Part they're in
CATEGORY_PART = {
    "Abuse": 1,
    "Arrest": 1,
    "Arson": 1,
    "Assault": 1,
    "Burglary": 2,
    "Explosion": 2,
    "Fighting": 2,
    "RoadAccidents": 3,
    "Robbery": 3,
    "Shooting": 3,
    "Shoplifting": 4,
    "Stealing": 4,
    "Vandalism": 4,
}


def find_video(video_dir: Path, annotation_name: str, is_abnormal: bool) -> Path | None:
    """Map annotation filename to video file.

    Abnormal: Abuse002_x264_18.json → Abuse002_x264.mp4
              in Anomaly-Videos-Part-N/Category/
    Normal:   Abuse002_x264_15.json → Abuse002_x264.mp4
              could be in any Part or Testing_Normal_Videos_Anomaly/
    """
    # Strip segment index: Abuse002_x264_18 → Abuse002_x264
    stem = annotation_name.replace(".json", "")
    # Remove last _N segment index
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        video_stem = parts[0]
    else:
        video_stem = stem

    video_name = f"{video_stem}.mp4"

    if is_abnormal:
        # Extract category from name (e.g., "Abuse" from "Abuse002_x264")
        category = re.match(r"([A-Za-z]+)", video_stem)
        if category:
            cat = category.group(1)
            part_num = CATEGORY_PART.get(cat)
            if part_num:
                path = video_dir / f"Anomaly-Videos-Part-{part_num}" / cat / video_name
                if path.exists():
                    return path
        # Fallback: search all parts
        for part in ANOMALY_PARTS:
            for cat_dir in (video_dir / part).iterdir():
                if cat_dir.is_dir():
                    path = cat_dir / video_name
                    if path.exists():
                        return path
    else:
        # Normal videos: check Testing_Normal_Videos_Anomaly/ first
        # Normal annotation names map to videos that could be anywhere
        normal_dir = video_dir / "Testing_Normal_Videos_Anomaly"
        if normal_dir.exists():
            # Normal videos have format: Normal_Videos_NNN_x264.mp4
            # But annotation names like Abuse002_x264_15.json → video is Abuse002_x264.mp4
            # These are normal clips from abnormal-category videos
            path = normal_dir / video_name
            if path.exists():
                return path
        # Also check anomaly dirs (some normal clips come from anomaly videos)
        for part in ANOMALY_PARTS:
            for cat_dir in (video_dir / part).iterdir():
                if cat_dir.is_dir():
                    path = cat_dir / video_name
                    if path.exists():
                        return path

    return None


def extract_frame(video_path: Path, position: float = 0.5) -> Image.Image | None:
    """Extract a single frame from video at given position (0.0-1.0)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    target_frame = int(total_frames * position)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # BGR → RGB → PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def extract_multi_frames(video_path: Path, n_frames: int = 8) -> list[Image.Image]:
    """Extract N evenly-spaced frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Evenly spaced positions
    positions = [int(i * total_frames / (n_frames + 1)) for i in range(1, n_frames + 1)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def build_samples(
    annotation_dir: Path,
    video_dir: Path,
    frames_per_clip: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Build training samples from annotations + videos."""

    random.seed(seed)

    abnormal_dir = annotation_dir / "UCF_abnormal"
    normal_dir = annotation_dir / "UCF_normal"

    abnormal_files = sorted(abnormal_dir.glob("*.json"))
    normal_files = sorted(normal_dir.glob("*.json"))

    print(f"Found {len(abnormal_files)} abnormal, {len(normal_files)} normal annotations")

    # Balance: undersample normal to match abnormal
    if len(normal_files) > len(abnormal_files):
        random.shuffle(normal_files)
        normal_files = normal_files[: len(abnormal_files)]
        print(f"Balanced to {len(abnormal_files)} each")

    # If max_samples, take half from each
    if max_samples:
        half = max_samples // 2
        abnormal_files = abnormal_files[:half]
        normal_files = normal_files[:half]
        print(
            f"Limited to {len(abnormal_files)} abnormal + {len(normal_files)} normal = {len(abnormal_files) + len(normal_files)}"
        )

    samples = []
    skipped = 0

    # All QA pair types in annotations (richer training signal)
    QA_TYPES = [
        "detection_qa_pairs",
        "classification_qa_pairs",
        "subject_qa_pairs",
        "description_qa_pairs",
        "cause_qa_pairs",
        "temporal_qa_pairs",
        "spatial_qa_pairs",
        "consequence_qa_pairs",
        "prevention_qa_pairs",
        "contextual_qa_pairs",
        "counterfactual_qa_pairs",
        "ethical_qa_pairs",
    ]

    # Process abnormal clips — extract ALL QA types for richer training
    for ann_file in abnormal_files:
        video_path = find_video(video_dir, ann_file.name, is_abnormal=True)
        if video_path is None:
            skipped += 1
            continue

        if frames_per_clip == 1:
            frame = extract_frame(video_path)
            if frame is None:
                skipped += 1
                continue
            frames = [frame]
        else:
            frames = extract_multi_frames(video_path, frames_per_clip)
            if not frames:
                skipped += 1
                continue

        # Load QA
        with open(ann_file) as f:
            qa = json.load(f)

        # Extract from all available QA types
        has_any = False
        for qa_type in QA_TYPES:
            pairs = qa.get(qa_type, [])
            for pair in pairs:
                q, a = pair.get("Q", ""), pair.get("A", "")
                if not q or not a:
                    continue
                # Some answers are lists — join them
                if isinstance(a, list):
                    a = ", ".join(str(x) for x in a)
                a = str(a)
                has_any = True
                samples.append(
                    {
                        "images": frames,
                        "question": q,
                        "answer": a,
                        "source": ann_file.stem,
                        "label": "abnormal",
                        "qa_type": qa_type,
                    }
                )

        if not has_any:
            skipped += 1

    # Process normal clips — synthesize detection QA
    # Normal annotations don't have detection_qa_pairs, so we create them
    detection_questions = [
        "Does this video contain any potentially violent or criminal activities?",
        "Is there any abnormal or suspicious activity in this surveillance footage?",
        "Does this surveillance video show any dangerous or illegal behavior?",
        "Are there any security concerns or unusual events visible in this video?",
    ]

    for ann_file in normal_files:
        video_path = find_video(video_dir, ann_file.name, is_abnormal=False)
        if video_path is None:
            skipped += 1
            continue

        if frames_per_clip == 1:
            frame = extract_frame(video_path)
            if frame is None:
                skipped += 1
                continue
            frames = [frame]
        else:
            frames = extract_multi_frames(video_path, frames_per_clip)
            if not frames:
                skipped += 1
                continue

        question = random.choice(detection_questions)

        samples.append(
            {
                "images": frames,
                "question": question,
                "answer": "No. The surveillance footage shows normal activity with no signs of violence, criminal behavior, or security threats.",
                "source": ann_file.stem,
                "label": "normal",
                "qa_type": "detection_qa_pairs",
            }
        )

    print(f"Built {len(samples)} samples ({skipped} skipped — video not found or unreadable)")

    # Shuffle
    random.shuffle(samples)
    return samples


def to_hf_dataset(samples: list[dict], output_dir: Path):
    """Convert samples to HuggingFace dataset format and save."""

    # mlx-vlm expects: images (list of PIL), messages (chat format)
    hf_samples = []

    for s in samples:
        hf_samples.append(
            {
                "images": s["images"],  # PIL images, saved to disk later
                "question": s["question"],
                "answer": s["answer"],
                "qa_type": s.get("qa_type", "detection_qa_pairs"),
            }
        )

    # Split: 90% train, 10% validation
    n_val = max(1, len(hf_samples) // 10)
    train_samples = hf_samples[n_val:]
    val_samples = hf_samples[:n_val]

    # Save as individual JSON files (mlx-vlm compatible)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images as files + create JSONL
    train_dir = output_dir / "train"
    val_dir = output_dir / "valid"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    def save_split(split_samples, split_dir, split_name):
        records = []
        for i, sample in enumerate(split_samples):
            # Save image(s) to disk
            image_paths = []
            for j, img in enumerate(sample["images"]):
                img_path = split_dir / f"{i:05d}_{j}.jpg"
                img.save(img_path, quality=85)
                # Use absolute path for mlx-vlm
                image_paths.append(str(img_path.resolve()))

            # Pre-build messages in Qwen VLM format.
            # Use JSON-string messages to avoid PyArrow mixed-type issues.
            question = sample["question"]
            if sample.get("qa_type") == "detection_qa_pairs":
                question += "\nAnswer Yes or No, followed by a brief reason."
            question += "\nBe concise."

            # Store as question/answer/image — simple flat columns
            records.append(
                {
                    "question": question,
                    "answer": sample["answer"],
                    "image": image_paths[0],
                }
            )

        # Save as JSONL
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"  {split_name}: {len(records)} samples → {jsonl_path}")

    save_split(train_samples, train_dir, "train")
    save_split(val_samples, val_dir, "valid")

    # Also save a config file
    config = {
        "total_samples": len(hf_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "abnormal_count": sum(1 for s in samples if s["label"] == "abnormal"),
        "normal_count": sum(1 for s in samples if s["label"] == "normal"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {output_dir / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SurveillanceVQA data for LoRA")
    parser.add_argument("--video-dir", default="surveillance_vqa/videos", help="Video directory")
    parser.add_argument(
        "--annotation-dir", default="surveillance_vqa/test_datasets", help="Annotation directory"
    )
    parser.add_argument(
        "--output-dir", default="surveillance_vqa/lora_dataset", help="Output directory"
    )
    parser.add_argument("--frames-per-clip", type=int, default=1, help="Frames to extract per clip")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples (for smoke test)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)

    print(f"Video dir: {video_dir}")
    print(f"Annotation dir: {annotation_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Frames per clip: {args.frames_per_clip}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print()

    samples = build_samples(
        annotation_dir,
        video_dir,
        frames_per_clip=args.frames_per_clip,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if not samples:
        print("ERROR: No samples built!")
        return

    print()
    to_hf_dataset(samples, output_dir)
    print(f"\nDone! Dataset ready at {output_dir}")


if __name__ == "__main__":
    main()
