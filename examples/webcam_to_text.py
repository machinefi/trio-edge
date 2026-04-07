#!/usr/bin/env python3
"""Webcam to Text — continuous VLM analysis from your laptop camera.

Usage:
    python examples/webcam_to_text.py
    python examples/webcam_to_text.py --prompt "Is anyone wearing glasses?"
    python examples/webcam_to_text.py --stride 60 --prompt "What objects are on the desk?"

Requires:     pip install 'trio-core[mlx]' or pip install 'trio-core[cuda]'
"""

import argparse
import sys
import time

from trio_core import StreamCapture, TrioCore


def main():
    parser = argparse.ArgumentParser(description="Webcam to text via VLM")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--prompt",
        "-p",
        default="Describe what you see in one sentence.",
        help="Question to ask the VLM",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Process every N-th frame (default: 30, ~1 per second at 30fps)",
    )
    parser.add_argument("--model", "-m", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--backend", "-b", default=None, help="Force backend: mlx or transformers")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    args = parser.parse_args()

    # Initialize engine
    from trio_core import EngineConfig

    config_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        config_kwargs["model"] = args.model
    config = EngineConfig(**config_kwargs)

    engine = TrioCore(config, backend=args.backend)
    print(f"Loading model: {config.model}")
    engine.load()

    health = engine.health()
    print(f"Backend: {health.get('backend', {}).get('backend', 'unknown')}")
    print(f"Device: {health.get('backend', {}).get('device', 'unknown')}")
    print(f"Profile: {health['profile']['family']} {health['profile']['param_size']}")
    print(f"\nWatching camera {args.camera} (stride={args.stride})")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    # Stream from webcam
    frame_num = 0
    try:
        with StreamCapture(str(args.camera), vid_stride=args.stride) as cap:
            for frame in cap:
                frame_num += 1
                t0 = time.monotonic()
                result = engine.analyze_frame(frame, args.prompt, max_tokens=args.max_tokens)
                elapsed = (time.monotonic() - t0) * 1000

                print(f"\n[Frame {frame_num}] ({elapsed:.0f}ms)")
                print(f"  {result.text}")
                print(
                    f"  Preprocess: {result.metrics.preprocess_ms:.0f}ms | "
                    f"Inference: {result.metrics.inference_ms:.0f}ms | "
                    f"Tokens/s: {result.metrics.tokens_per_sec:.1f}"
                )
    except KeyboardInterrupt:
        print(f"\nStopped after {frame_num} frames.")
    except IOError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        print(
            "Tip: Make sure a camera is connected (Mac Studio needs an external webcam).",
            file=sys.stderr,
        )
        print("     You can also use video_analyze.py for file-based analysis.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
