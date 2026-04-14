#!/usr/bin/env python3
"""Run eval suite and save baseline metrics.

Usage:
    python examples/run_eval.py
    python examples/run_eval.py --output results/baseline.json --runs 5
    python examples/run_eval.py --compress 0.5 --output eval_compressed_50.json
    python examples/run_eval.py --compare eval_baseline.json eval_compressed_50.json
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TrioCore eval suite")
    parser.add_argument(
        "--output",
        "-o",
        default="eval_baseline.json",
        help="Output JSON path (default: eval_baseline.json)",
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=3, help="Number of runs per test case (default: 3)"
    )
    parser.add_argument("--model", "-m", default=None, help="Model name (default: auto)")
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="Max generation tokens (default: 64)"
    )
    parser.add_argument(
        "--compress",
        type=float,
        default=None,
        help="Enable token compression with given ratio (e.g. 0.5 = keep 50%%)",
    )
    parser.add_argument(
        "--strategy",
        default="similarity",
        choices=["uniform", "similarity", "attention"],
        help="Compression strategy (default: similarity)",
    )
    parser.add_argument(
        "--tome",
        type=int,
        default=None,
        metavar="R",
        help="Enable ToMe with r tokens merged per layer (e.g. --tome 8)",
    )
    parser.add_argument(
        "--metric",
        default="keys",
        choices=["keys", "hidden"],
        help="ToMe similarity metric (default: keys)",
    )
    parser.add_argument(
        "--min-keep", type=float, default=0.3, help="Min fraction of tokens to keep (default: 0.3)"
    )
    parser.add_argument(
        "--resolution",
        default="480p",
        choices=["480p", "720p", "1080p"],
        help="Input resolution (default: 480p)",
    )
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two saved reports")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        from trio_core.eval import EvalReport

        EvalReport.compare(args.compare[0], args.compare[1])
        return

    # Run eval
    from trio_core import TrioCore
    from trio_core.config import EngineConfig
    from trio_core.eval import EvalSuite

    config_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        config_kwargs["model"] = args.model
    # Disable dedup and motion gate for clean measurements
    config_kwargs["dedup_enabled"] = False
    config_kwargs["motion_enabled"] = False

    if args.tome:
        config_kwargs["tome_enabled"] = True
        config_kwargs["tome_r"] = args.tome
        config_kwargs["tome_metric"] = args.metric
        config_kwargs["tome_min_keep_ratio"] = args.min_keep

    config = EngineConfig(**config_kwargs)
    engine = TrioCore(config)

    if args.tome:
        print(f"ToMe: r={args.tome}, metric={args.metric}, min_keep={args.min_keep}")
        print(f"Loading model: {config.model}")
        engine.load()
    elif args.compress:
        # Use compressed backend — load manually to bypass auto_backend
        from trio_core.backends.compressed import CompressedMLXBackend
        from trio_core.token_compression import TokenCompressor

        compressor = TokenCompressor(strategy=args.strategy, ratio=args.compress)
        backend = CompressedMLXBackend(config.model, compressor)
        print(f"Compression: ratio={args.compress}, strategy={args.strategy}")
        print(f"Loading model: {config.model}")
        backend.load()
        engine._backend = backend
        engine._loaded = True
    else:
        print(f"Loading model: {config.model}")
        engine.load()

    health = engine.health()
    print(f"Backend: {health['backend']['backend']}")
    print(f"Device: {health['backend']['device']}")
    print(f"Profile: {health['profile']['family']} {health['profile']['param_size']}")
    print(f"\nRunning eval ({args.runs} runs per case)...")
    print("-" * 60)

    resolution_map = {"480p": (480, 640), "720p": (720, 1280), "1080p": (1080, 1920)}
    res_h, res_w = resolution_map[args.resolution]
    print(f"Resolution: {args.resolution} ({res_w}x{res_h})")

    suite = EvalSuite(
        engine, n_runs=args.runs, max_tokens=args.max_tokens, height=res_h, width=res_w
    )
    report = suite.run()
    report.print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)


if __name__ == "__main__":
    main()
