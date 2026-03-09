#!/usr/bin/env python3
"""Evaluate LoRA adapter on validation set.

Usage:
    source .venv-lora/bin/activate
    python scripts/eval_lora_adapter.py --adapter-path adapters/surveillance-qwen35-2b
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image_processor, load_config


def normalize(text: str) -> str:
    """Normalize answer for comparison."""
    # Strip think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Take first line only
    text = text.split("\n")[0].strip()
    return text.lower()


def is_correct(predicted: str, expected: str, question: str) -> bool:
    """Check if prediction matches expected answer."""
    pred = normalize(predicted)
    exp = normalize(expected)

    # For yes/no questions
    if "answer yes or no" in question.lower():
        pred_yn = "yes" if pred.startswith("yes") else "no" if pred.startswith("no") else pred
        exp_yn = "yes" if exp.startswith("yes") else "no" if exp.startswith("no") else exp
        return pred_yn == exp_yn

    # For short answers (classification, subject)
    if len(exp.split()) <= 5:
        return exp in pred or pred in exp

    # For long-form, check key terms overlap
    exp_words = set(exp.split())
    pred_words = set(pred.split())
    if len(exp_words) == 0:
        return False
    overlap = len(exp_words & pred_words) / len(exp_words)
    return overlap > 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="mlx-community/Qwen3.5-2B-4bit")
    parser.add_argument("--adapter-path", default="adapters/surveillance-qwen35-2b")
    parser.add_argument("--val-jsonl", default="surveillance_vqa/lora_dataset/jsonl/valid.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--run-baseline", action="store_true", help="Also run without adapter for comparison")
    args = parser.parse_args()

    # Load validation data
    val_data = []
    with open(args.val_jsonl) as f:
        for line in f:
            val_data.append(json.loads(line))
    if args.max_samples:
        val_data = val_data[:args.max_samples]
    print(f"Loaded {len(val_data)} validation samples")

    configs_to_run = [("lora", args.adapter_path)]
    if args.run_baseline:
        configs_to_run.append(("baseline", None))

    for config_name, adapter_path in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")

        # Load model
        if adapter_path:
            model, processor = load(args.model_path, adapter_path=adapter_path)
        else:
            model, processor = load(args.model_path)

        config = load_config(args.model_path)

        correct = 0
        total = 0
        empty = 0
        results_by_type = {}

        for i, sample in enumerate(val_data):
            question = sample["question"]
            expected = sample["answer"]
            image_path = sample["image"]

            # Generate
            prompt = apply_chat_template(processor, config, question, num_images=1)
            output = generate(
                model, processor, prompt,
                image=[image_path],
                max_tokens=args.max_tokens,
                verbose=False,
            )

            # Handle GenerationResult
            if hasattr(output, "text"):
                pred_text = output.text
            else:
                pred_text = str(output)

            if not pred_text.strip():
                empty += 1

            match = is_correct(pred_text, expected, question)
            if match:
                correct += 1
            total += 1

            # Track by QA type (infer from question)
            if "answer yes or no" in question.lower():
                qa_type = "detection"
            elif "type of abnormal" in question.lower():
                qa_type = "classification"
            elif "who or what" in question.lower():
                qa_type = "subject"
            elif "what is happening" in question.lower() or "describe" in question.lower():
                qa_type = "description"
            elif "what led to" in question.lower():
                qa_type = "cause"
            else:
                qa_type = "other"

            if qa_type not in results_by_type:
                results_by_type[qa_type] = {"correct": 0, "total": 0}
            results_by_type[qa_type]["total"] += 1
            if match:
                results_by_type[qa_type]["correct"] += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(val_data)}] acc={correct/total:.1%} empty={empty}")

        print(f"\n--- {config_name} Results ---")
        print(f"Overall: {correct}/{total} = {correct/total:.1%} (empty={empty})")
        for qt, res in sorted(results_by_type.items()):
            acc = res["correct"] / res["total"] if res["total"] > 0 else 0
            print(f"  {qt}: {res['correct']}/{res['total']} = {acc:.1%}")

        # Cleanup
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()
