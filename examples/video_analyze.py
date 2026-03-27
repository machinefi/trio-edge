#!/usr/bin/env python3
"""Video Analyze — analyze a video file with detailed metrics.

Usage:
    python examples/video_analyze.py video.mp4
    python examples/video_analyze.py video.mp4 --prompt "Count the people visible"
    python examples/video_analyze.py video.mp4 --no-dedup --fps 4

Requires: pip install 'trio-core[mlx]' or pip install 'trio-core[transformers]'
"""

import argparse
import time

from trio_core import EngineConfig, TrioCore, get_profile


def main():
    parser = argparse.ArgumentParser(description="Analyze a video file with VLM")
    parser.add_argument("video", help="Path to video file or URL")
    parser.add_argument(
        "--prompt",
        "-p",
        default="Describe what is happening in this video.",
        help="Question to ask",
    )
    parser.add_argument("--model", "-m", default=None, help="Model name")
    parser.add_argument("--backend", "-b", default=None, help="Force backend: mlx or transformers")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS for frame extraction")
    parser.add_argument("--max-frames", type=int, default=128, help="Max frames to extract")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--no-dedup", action="store_true", help="Disable temporal deduplication")
    parser.add_argument("--motion", action="store_true", help="Enable motion gating")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed metrics")
    args = parser.parse_args()

    # Configure
    config_kwargs = {
        "video_fps": args.fps,
        "video_max_frames": args.max_frames,
        "max_tokens": args.max_tokens,
        "dedup_enabled": not args.no_dedup,
        "motion_enabled": args.motion,
    }
    if args.model:
        config_kwargs["model"] = args.model
    config = EngineConfig(**config_kwargs)

    # Load
    engine = TrioCore(config, backend=args.backend)
    print(f"Loading model: {config.model}")
    engine.load()

    profile = get_profile(config.model)
    print(
        f"Profile: {profile.family} {profile.param_size} "
        f"(merge_factor={profile.merge_factor}, max_visual_tokens={profile.max_visual_tokens})"
    )
    print(f"Analyzing: {args.video}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    # Analyze
    t0 = time.monotonic()
    result = engine.analyze_video(args.video, args.prompt)
    total = (time.monotonic() - t0) * 1000

    # Output
    print(f"\n{result.text}\n")
    print("-" * 60)

    m = result.metrics
    print(f"Total latency:     {total:.0f}ms")
    print(f"  Preprocess:      {m.preprocess_ms:.0f}ms")
    print(f"  Inference:       {m.inference_ms:.0f}ms")
    print(f"  Postprocess:     {m.postprocess_ms:.0f}ms")
    print(f"Frames:            {m.frames_input} input -> {m.frames_after_dedup} after dedup")
    print(f"Dedup removed:     {m.dedup_removed} frames")

    if args.verbose:
        print(f"Motion skipped:    {m.motion_skipped}")
        print(f"Prompt tokens:     {m.prompt_tokens}")
        print(f"Completion tokens: {m.completion_tokens}")
        print(f"Tokens/sec:        {m.tokens_per_sec:.1f}")
        print(f"Peak memory:       {m.peak_memory_gb:.1f} GB")


if __name__ == "__main__":
    main()
