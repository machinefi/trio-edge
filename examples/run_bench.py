#!/usr/bin/env python3
"""Unified benchmark CLI for TrioCore.

Usage:
    # Quick single run
    python examples/run_bench.py --models qwen2.5-vl-3b --benchmarks pope -n 50

    # Full sweep
    python examples/run_bench.py --benchmarks pope,gqa --configs baseline,tome_r4,compressed_50 -n 100

    # Preview what will run
    python examples/run_bench.py --benchmarks pope --configs baseline,tome_r4 --dry-run

    # Skip already-completed combos
    python examples/run_bench.py --benchmarks pope --skip-existing

    # Filter by family
    python examples/run_bench.py --families qwen3-vl --benchmarks pope -n 100

    # Filter by tier
    python examples/run_bench.py --tier 1 --benchmarks pope -n 50

    # Reports
    python examples/run_bench.py --report model_x_config --benchmark pope
    python examples/run_bench.py --report delta --benchmark pope --baseline baseline --compare tome_r4

    # List available models/benchmarks/configs
    python examples/run_bench.py --list models
    python examples/run_bench.py --list benchmarks
    python examples/run_bench.py --list configs
"""

import argparse
import sys


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_list(args):
    from trio_core.bench import MODEL_REGISTRY, OPTIM_PRESETS, BENCHMARKS

    if args.list == "models":
        print(f"\n{'Key':<20} {'Family':<14} {'Tier':<6} {'HF ID'}")
        print("-" * 80)
        for entry in MODEL_REGISTRY.values():
            print(f"{entry.key:<20} {entry.family:<14} T{entry.tier:<5} {entry.hf_id}")
            caps = []
            if entry.supports_tome:
                caps.append("tome")
            if entry.supports_kv_reuse:
                caps.append("kv_reuse")
            print(f"{'':>20} caps: {', '.join(caps)}")

    elif args.list == "benchmarks":
        print("\nAvailable benchmarks:")
        for b in BENCHMARKS:
            print(f"  {b}")

    elif args.list == "configs":
        print(f"\n{'Name':<20} {'Requires':<30} Description")
        print("-" * 80)
        for cfg in OPTIM_PRESETS.values():
            reqs = ", ".join(cfg.requires) if cfg.requires else "—"
            print(f"{cfg.name:<20} {reqs:<30} {cfg.description}")


def cmd_report(args):
    from trio_core.bench import ReportGenerator, ResultStore

    store = ResultStore(args.results_dir)
    gen = ReportGenerator(store)

    if args.report == "model_x_config":
        if not args.benchmark:
            print("--benchmark required for model_x_config report")
            sys.exit(1)
        print(gen.model_x_config(args.benchmark, metric=args.metric or "accuracy"))

    elif args.report == "model_x_benchmark":
        cfg = args.baseline or "baseline"
        print(gen.model_x_benchmark(cfg, metric=args.metric or "accuracy"))

    elif args.report == "delta":
        if not all([args.benchmark, args.baseline, args.compare]):
            print("--benchmark, --baseline, and --compare required for delta report")
            sys.exit(1)
        print(gen.delta_report(args.benchmark, args.baseline, args.compare))


def cmd_run(args):
    from pathlib import Path
    from trio_core.bench import BenchSweep, get_models, OPTIM_PRESETS

    # Resolve models
    model_keys = parse_csv(args.models)
    families = parse_csv(args.families)
    tier = args.tier

    models = get_models(keys=model_keys, families=families, tier=tier)
    if not models:
        print("No models matched filters.")
        sys.exit(1)

    # Resolve configs
    config_names = parse_csv(args.configs) or ["baseline"]
    for name in config_names:
        if name not in OPTIM_PRESETS:
            print(f"Unknown config: {name!r}. Use --list configs to see available.")
            sys.exit(1)

    # Resolve benchmarks
    benchmark_names = parse_csv(args.benchmarks) or ["pope"]

    # Benchmark-specific kwargs
    bm_kwargs = {}
    if args.mvbench_dir:
        bm_kwargs["mvbench_dir"] = args.mvbench_dir
    if args.mvbench_tasks:
        bm_kwargs["mvbench_tasks"] = parse_csv(args.mvbench_tasks)
    if args.split:
        bm_kwargs["split"] = args.split
    if args.surveillance_dir:
        bm_kwargs["surveillance_dir"] = args.surveillance_dir
    if args.surveillance_video_dir:
        bm_kwargs["surveillance_video_dir"] = args.surveillance_video_dir
    if args.surveillance_qa_type:
        bm_kwargs["surveillance_qa_type"] = args.surveillance_qa_type
    if args.surveillance_sources:
        bm_kwargs["surveillance_sources"] = parse_csv(args.surveillance_sources)

    sweep = BenchSweep(
        models=models,
        benchmarks=benchmark_names,
        configs=config_names,
        max_samples=args.n,
        max_tokens=args.max_tokens,
        skip_existing=args.skip_existing,
        results_dir=Path(args.results_dir),
        benchmark_kwargs=bm_kwargs,
        adapter_path=args.adapter_path,
    )

    if args.dry_run:
        sweep.plan()
        return

    records = sweep.run()

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {len(records)} runs")
    print(f"{'='*60}")
    for r in records:
        flag = " !!!" if r.anomalies else ""
        print(f"  {r.model_key:<20} {r.benchmark:<10} {r.config:<20} acc={r.accuracy:.1%}{flag}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified TrioCore benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # List mode
    parser.add_argument("--list", choices=["models", "benchmarks", "configs"],
                        help="List available models, benchmarks, or configs")

    # Report mode
    parser.add_argument("--report", choices=["model_x_config", "model_x_benchmark", "delta"],
                        help="Generate a comparison report")
    parser.add_argument("--benchmark", help="Benchmark name (for reports)")
    parser.add_argument("--baseline", help="Baseline config (for delta report)")
    parser.add_argument("--compare", help="Compare config (for delta report)")
    parser.add_argument("--metric", help="Metric for reports (default: accuracy)")

    # Run mode — model filters
    parser.add_argument("--models", help="Comma-separated model keys")
    parser.add_argument("--families", help="Comma-separated model families")
    parser.add_argument("--tier", type=int, help="Filter by tier (1 or 2)")

    # Run mode — benchmark/config
    parser.add_argument("--benchmarks", help="Comma-separated benchmark names (default: pope)")
    parser.add_argument("--configs", help="Comma-separated config names (default: baseline)")
    parser.add_argument("-n", type=int, default=None, help="Max samples per benchmark")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max generation tokens (default: 16)")

    # Run mode — options
    parser.add_argument("--dry-run", action="store_true", help="Preview combos without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip combos with existing results")
    parser.add_argument("--results-dir", default="research/bench-results", help="Results directory")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter directory")

    # Benchmark-specific
    parser.add_argument("--split", default=None, help="POPE split (random/popular/adversarial)")
    parser.add_argument("--mvbench-dir", default=None, help="MVBench video directory")
    parser.add_argument("--mvbench-tasks", default=None, help="Comma-separated MVBench tasks")
    parser.add_argument("--surveillance-dir", default=None, help="SurveillanceVQA data directory")
    parser.add_argument("--surveillance-video-dir", default=None, help="SurveillanceVQA video directory")
    parser.add_argument("--surveillance-qa-type", default=None,
                        help="SurveillanceVQA QA type (detection/classification/description/all_abnormal/all)")
    parser.add_argument("--surveillance-sources", default=None,
                        help="Comma-separated video sources (UCF/MSAD/MEVA/NWPU_Test)")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.report:
        cmd_report(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
