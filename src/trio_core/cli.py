"""CLI for TrioCore — serve, analyze, bench, and device detection."""

from __future__ import annotations

import json
import logging

import typer

app = typer.Typer(
    name="trio-core",
    help="Portable video inference engine for VLMs — Apple Silicon, NVIDIA, and CPU",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def device():
    """Detect hardware and show recommended backend + model."""
    from trio_core.device import detect_device, recommend_model

    info = detect_device()
    model = recommend_model(info)

    typer.echo(f"Device:      {info.device_name}")
    typer.echo(f"Accelerator: {info.accelerator}")
    typer.echo(f"Memory:      {info.memory_gb:.1f} GB")
    typer.echo(f"GPU cores:   {info.compute_units or 'N/A'}")
    typer.echo(f"Backend:     {info.backend}")
    typer.echo(f"Model:       {model}")

    # Check backend availability
    if info.backend == "mlx":
        try:
            import mlx.core  # noqa: F401
            typer.echo("Status:      mlx-vlm installed")
        except ImportError:
            typer.echo("Status:      mlx-vlm NOT installed — run: pip install 'trio-core[mlx]'")
    else:
        try:
            import torch  # noqa: F401
            typer.echo(f"Status:      torch installed (CUDA={torch.cuda.is_available()})")
        except ImportError:
            typer.echo("Status:      torch NOT installed — run: pip install 'trio-core[transformers]'")


@app.command()
def serve(
    model: str = typer.Option(None, "--model", "-m", help="HuggingFace model ID or local path"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Start the TrioCore API server."""
    _setup_logging(verbose)
    import uvicorn
    from trio_core.config import EngineConfig
    from trio_core.api.server import create_app

    config = EngineConfig()
    if model:
        config.model = model
    config.host = host
    config.port = port

    app_instance = create_app(config, backend=backend)
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Video file path or URL"),
    prompt: str = typer.Option(
        "Describe what is happening in this video.",
        "--prompt", "-q",
        help="Question or instruction",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max generation tokens"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON with metrics"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Analyze a video file with the VLM."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
        config.model = model

    engine = TrioCore(config, backend=backend)

    typer.echo(f"Loading model: {config.model}")
    engine.load()

    typer.echo(f"Analyzing: {video}")
    result = engine.analyze_video(
        video=video,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if json_output:
        output = {
            "text": result.text,
            "model": config.model,
            "backend": engine._backend.backend_name if engine._backend else "none",
            "metrics": result.metrics.__dict__,
        }
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"\n{result.text}")
        typer.echo(f"\n--- Metrics ---")
        m = result.metrics
        typer.echo(f"Backend: {engine._backend.backend_name if engine._backend else 'N/A'} ({engine._backend.device_info.device_name if engine._backend else 'N/A'})")
        typer.echo(f"Frames: {m.frames_input} → {m.frames_after_dedup} (dedup removed {m.dedup_removed})")
        typer.echo(f"Tokens: {m.prompt_tokens} prompt + {m.completion_tokens} completion")
        typer.echo(f"Speed: {m.tokens_per_sec:.1f} tok/s | Latency: {m.latency_ms:.0f}ms")
        typer.echo(f"Pipeline: preprocess={m.preprocess_ms:.0f}ms inference={m.inference_ms:.0f}ms postprocess={m.postprocess_ms:.0f}ms")
        if m.peak_memory_gb > 0:
            typer.echo(f"Memory: {m.peak_memory_gb:.2f} GB peak")


@app.command()
def bench(
    video: str = typer.Argument(..., help="Video file path"),
    prompt: str = typer.Option("Describe this video.", "--prompt", "-q"),
    model: str = typer.Option(None, "--model", "-m"),
    backend: str = typer.Option(None, "--backend", "-b"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Benchmark video inference."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
        config.model = model

    engine = TrioCore(config, backend=backend)
    typer.echo(f"Loading model: {config.model}")
    engine.load()
    typer.echo(f"Backend: {engine._backend.backend_name} ({engine._backend.device_info.device_name})")

    latencies = []
    for i in range(runs):
        result = engine.analyze_video(video=video, prompt=prompt)
        latencies.append(result.metrics.latency_ms)
        typer.echo(f"  Run {i + 1}/{runs}: {result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.1f} tok/s")

    avg = sum(latencies) / len(latencies)
    typer.echo(f"\nBenchmark: {runs} runs, avg {avg:.0f}ms")


if __name__ == "__main__":
    app()
