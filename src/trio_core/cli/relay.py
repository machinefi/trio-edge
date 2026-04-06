from __future__ import annotations

import shutil

import typer

from trio_core.cli._shared import _setup_logging, app
from trio_core.cli.cam import _resolve_rtsp_url
from trio_core.http_ingest_relay import HttpIngestRelay, RelayError


@app.command(help="HTTP MPEG-TS relay to Trio Cloud.")
def relay(
    cloud: str = typer.Option(
        "https://trio-relay.machinefi.com",
        "--cloud",
        help="Trio Cloud base URL (e.g. https://trio-relay.machinefi.com)",
    ),
    source: str = typer.Option(
        "0",
        "--camera",
        "--source",
        "-s",
        help="Video source: RTSP URL, camera index, or video file",
    ),
    camera_id: str = typer.Option(
        None, "--camera-id", help="Override the derived camera identifier"
    ),
    host: str = typer.Option(None, "--host", "-h", help="Camera IP address (skip discovery)"),
    port: int = typer.Option(None, "--port", help="ONVIF port"),
    user: str = typer.Option(None, "--user", "-u", help="Camera username"),
    password: str = typer.Option("", "--password", "-p", help="Camera password"),
    rtsp: str = typer.Option(None, "--rtsp", help="Direct RTSP URL (bypasses source)"),
    discover: bool = typer.Option(
        False, "--discover", help="Interactively discover ONVIF cameras to use as source"
    ),
    token: str = typer.Option(
        None, "--token", "-t", help="Bearer token for Trio Cloud authentication"
    ),
    resolution: str | None = typer.Option(
        None, "--resolution", "-r", help="Video resolution WxH (e.g. 1280x720)"
    ),
    framerate: int = typer.Option(30, "--framerate", "--fps", help="Target frame rate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    json_logs: bool = typer.Option(
        False, "--json-logs", help="Structured JSON logging (or set TRIO_LOG_JSON=1)"
    ),
):
    """Relay video from a webcam, RTSP camera, or file to Trio Cloud via HTTP MPEG-TS."""
    import asyncio

    _setup_logging(verbose, json_logs=json_logs)

    resolution_tuple = None
    if resolution:
        try:
            width, height = resolution.lower().split("x", 1)
            resolution_tuple = (int(width), int(height))
        except ValueError:
            typer.echo(
                f"✗ Invalid resolution format: {resolution} (expected WxH, e.g. 1280x720)",
                err=True,
            )
            raise typer.Exit(1)

    if not shutil.which("ffmpeg"):
        typer.echo(
            "✗ ffmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)",
            err=True,
        )
        raise typer.Exit(1)

    if not token:
        typer.echo("✗ Bearer token required. Pass --token <TOKEN>", err=True)
        raise typer.Exit(1)

    actual_source = source
    if discover or host or rtsp:
        actual_source = _resolve_rtsp_url(rtsp, host, port, user, password)

    relay_obj = HttpIngestRelay(
        source=actual_source,
        cloud_url=cloud,
        bearer_token=token,
        camera_id=camera_id,
        resolution=resolution_tuple,
        framerate=framerate,
    )

    typer.echo(f"Relay: {actual_source} -> {cloud}")
    typer.echo(f"Transport: HTTP MPEG-TS | FPS: {framerate} | Resolution: {resolution or 'native'}")
    typer.echo("Press Ctrl+C to stop.\n")

    async def _run() -> None:
        try:
            await relay_obj.run()
        finally:
            await relay_obj.teardown()

    try:
        asyncio.run(_run())
    except RelayError as exc:
        typer.echo(f"\n✗ {exc}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nStopping relay...")
        typer.echo("Disconnected.")
