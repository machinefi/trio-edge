from __future__ import annotations

import typer

from trio_core.cli._shared import app


@app.command()
def discover(
    timeout: int = typer.Option(5, "--timeout", "-t", help="Discovery timeout in seconds"),
):
    """Discover cameras on your network via ONVIF."""
    from trio_core.onvif import discover_cameras

    typer.echo("Searching for cameras on your network (ONVIF)...\n")

    cameras = discover_cameras(timeout=timeout)

    if not cameras:
        typer.echo("No cameras found. Make sure cameras are on the same subnet.")
        typer.echo("You can manually specify: trio relay --rtsp rtsp://admin:pass@IP/stream")
        raise typer.Exit(0)

    typer.echo(f"Found {len(cameras)} camera(s):\n")
    for i, c in enumerate(cameras):
        typer.echo(f"  [{i + 1}] {c.name} ({c.ip})")
        if c.rtsp_url:
            typer.echo(f"      RTSP: {c.rtsp_url}")
    typer.echo()
