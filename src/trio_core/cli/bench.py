from __future__ import annotations

import typer

from trio_core.cli._shared import _die_load_error, _require_gpu, _setup_logging, app


@app.command()
def bench(
    video: str = typer.Argument(..., help="Video file path"),
    prompt: str = typer.Option("Describe this video.", "--prompt", "-q"),
    model: str = typer.Option(None, "--model", "-m"),
    backend: str = typer.Option(None, "--backend", "-b"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Benchmark inference speed and hardware performance."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig.from_env_file()
    if not model:
        model, detected_backend = _require_gpu()
        if not backend:
            backend = detected_backend
    config.model = model

    engine = TrioCore(config, backend=backend)
    typer.echo(f"Loading {config.model}...")
    try:
        engine.load()
    except Exception as e:
        _die_load_error(e, config.model)
    typer.echo(f"{engine._backend.backend_name} ({engine._backend.device_info.device_name})")

    latencies = []
    for i in range(runs):
        result = engine.analyze_video(video=video, prompt=prompt)
        latencies.append(result.metrics.latency_ms)
        typer.echo(
            f"  Run {i + 1}/{runs}: {result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.1f} tok/s"
        )

    avg = sum(latencies) / len(latencies)
    typer.echo(f"\n{runs} runs, avg {avg:.0f}ms")
