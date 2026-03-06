#!/usr/bin/env python3
"""Run standard VLM benchmarks to measure quality and performance.

Usage:
    # POPE benchmark (object hallucination) — 100 samples, fast
    python examples/run_benchmark.py --bench pope --samples 100

    # POPE with ToMe compression
    python examples/run_benchmark.py --bench pope --samples 100 --tome 8

    # GQA benchmark (real-world visual reasoning)
    python examples/run_benchmark.py --bench gqa --samples 100

    # MMBench (multi-ability, 20 dimensions)
    python examples/run_benchmark.py --bench mmbench --samples 100

    # TextVQA benchmark
    python examples/run_benchmark.py --bench textvqa --samples 100

    # Compare two results
    python examples/run_benchmark.py --compare baseline.json tome_r8.json
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="VLM benchmark runner")
    parser.add_argument("--bench", choices=["pope", "textvqa", "gqa", "mmbench", "custom"],
                        default="pope", help="Benchmark to run")
    parser.add_argument("--split", default="random",
                        choices=["random", "popular", "adversarial"],
                        help="POPE split (default: random)")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Max samples (default: all)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (default: auto)")
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="Max generation tokens (default: 16)")
    parser.add_argument("--tome", type=int, default=None, metavar="R",
                        help="Enable ToMe with r tokens merged per layer")
    parser.add_argument("--metric", default="keys", choices=["keys", "hidden"],
                        help="ToMe similarity metric (default: keys)")
    parser.add_argument("--min-keep", type=float, default=0.3,
                        help="Min fraction of tokens to keep (default: 0.3)")
    parser.add_argument("--compress", type=float, default=None,
                        help="Post-encoder compression ratio")
    parser.add_argument("--custom-path", default=None,
                        help="Path to custom benchmark JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two saved benchmark results")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        from trio_core.eval_benchmarks import BenchmarkResult
        BenchmarkResult.compare(args.compare[0], args.compare[1])
        return

    # Default output path
    if args.output is None:
        suffix = ""
        if args.tome:
            suffix = f"_tome_r{args.tome}"
        elif args.compress:
            suffix = f"_compressed_{int(args.compress*100)}"
        else:
            suffix = "_baseline"
        n_str = f"_{args.samples}" if args.samples else ""
        args.output = f"research/eval-results/bench_{args.bench}_{args.split}{n_str}{suffix}.json"

    # Setup benchmark
    from trio_core.eval_benchmarks import (
        POPEBenchmark, TextVQABenchmark, GQABenchmark, MMBenchBenchmark,
        CustomBenchmark, BenchmarkRunner,
    )

    if args.bench == "pope":
        benchmark = POPEBenchmark(split=args.split, max_samples=args.samples)
    elif args.bench == "textvqa":
        benchmark = TextVQABenchmark(max_samples=args.samples)
    elif args.bench == "gqa":
        benchmark = GQABenchmark(max_samples=args.samples)
    elif args.bench == "mmbench":
        benchmark = MMBenchBenchmark(max_samples=args.samples)
    elif args.bench == "custom":
        if not args.custom_path:
            parser.error("--custom-path required for custom benchmark")
        benchmark = CustomBenchmark(args.custom_path)

    print(f"Benchmark: {benchmark.name}")

    # Setup engine
    from trio_core import TrioCore
    from trio_core.config import EngineConfig

    config_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        config_kwargs["model"] = args.model
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
        from trio_core.token_compression import TokenCompressor
        from trio_core.compressed_backend import CompressedMLXBackend
        compressor = TokenCompressor(strategy="similarity", ratio=args.compress)
        backend = CompressedMLXBackend(config.model, compressor)
        print(f"Compression: {args.compress}")
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
    print()

    # Run
    runner = BenchmarkRunner(engine, max_tokens=args.max_tokens)
    result = runner.run(benchmark)
    result.print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
