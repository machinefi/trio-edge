from __future__ import annotations

import json

import typer

from trio_core.cli._shared import _die_load_error, _require_gpu, _setup_logging, app


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Image or video file path"),
    prompt: str = typer.Option(
        "Describe what you see.",
        "--prompt",
        "-q",
        help="Question or instruction",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    adapter: str = typer.Option(None, "--adapter", "-a", help="LoRA adapter directory path"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max generation tokens"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output with metrics"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Run a natural language query against an image or video file."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig.from_env_file()
    if not model:
        model, detected_backend = _require_gpu()
        if not backend:
            backend = detected_backend
    config.model = model
    if adapter:
        config.adapter_path = adapter

    engine = TrioCore(config, backend=backend)

    if not json_output:
        typer.echo(f"Loading {config.model}...")
    try:
        engine.load()
    except Exception as e:
        _die_load_error(e, config.model)

    if not json_output:
        typer.echo(f"Analyzing {video}...")
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
        m = result.metrics
        typer.echo(
            f"\n{m.latency_ms:.0f}ms | {m.tokens_per_sec:.1f} tok/s | {m.prompt_tokens}+{m.completion_tokens} tokens"
        )
