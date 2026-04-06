from __future__ import annotations

import typer

from trio_core.cli._shared import app


@app.command()
def device():
    """Show hardware info and recommended model."""
    from trio_core.device import detect_device, recommend_model

    info = detect_device()
    model = recommend_model(info)

    typer.echo(f"Device:      {info.device_name}")
    typer.echo(f"Accelerator: {info.accelerator}")
    typer.echo(f"Memory:      {info.memory_gb:.1f} GB")
    typer.echo(f"GPU cores:   {info.compute_units or 'N/A'}")
    typer.echo(f"Backend:     {info.backend}")
    typer.echo(f"Model:       {model}")
