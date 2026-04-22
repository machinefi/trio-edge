from __future__ import annotations

import shutil

import typer

from trio_core.cli._shared import _setup_logging, app
from trio_core.cli.cam import _resolve_rtsp_url
from trio_core.http_ingest_relay import HttpIngestRelay, RelayError


@app.command(help="Stream a video feed to Trio Cloud.")
def relay(
    cloud: str = typer.Option(
        "https://trio-relay.machinefi.com",
        "--cloud",
        help="Trio Cloud base URL (e.g. https://trio-relay.machinefi.com)",
    ),
    source: str = typer.Option(
        "0",
        "--source",
        "-s",
        help="Video source: RTSP URL, camera index, or video file",
    ),
    camera_id: str = typer.Option(
        None, "--camera-id", help="Override the derived camera identifier"
    ),
    host: str = typer.Option(None, "--host", help="Camera IP address (skip discovery)"),
    port: int = typer.Option(None, "--port", help="ONVIF port"),
    user: str = typer.Option(None, "--user", "-u", help="Camera username"),
    password: str = typer.Option("", "--password", "-p", help="Camera password"),
    discover: bool = typer.Option(
        False, "--discover", help="Interactively discover ONVIF cameras to use as source"
    ),
    token: str = typer.Option(
        None, "--token", "-t", help="Bearer token for Trio Cloud authentication"
    ),
    resolution: str | None = typer.Option(
        None, "--resolution", "-r", help="Video resolution WxH (e.g. 1280x720)"
    ),
    fps: int = typer.Option(1, "--fps", help="Target frame rate"),
    segment_duration: float = typer.Option(
        10.0, "--segment-duration", help="Segment duration in seconds for each upload chunk"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show raw FFmpeg stderr output"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit NDJSON progress to stderr for machine consumption"
    ),
):
    """Relay video from a webcam, RTSP camera, or file to Trio Cloud via HTTP MPEG-TS."""
    import asyncio

    _setup_logging(verbose)

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
    if discover or host:
        actual_source = _resolve_rtsp_url(host, port, user, password)

    relay_obj = HttpIngestRelay(
        source=actual_source,
        cloud_url=cloud,
        bearer_token=token,
        camera_id=camera_id,
        resolution=resolution_tuple,
        framerate=fps,
        verbose=verbose,
        segment_duration=segment_duration,
        json_mode=json_output,
    )

    if not json_output:
        typer.echo(f"Relay: {actual_source} -> {cloud}")
        typer.echo(
            f"Transport: HTTP MPEG-TS (segmented {segment_duration}s) | FPS: {fps} | Resolution: {resolution or 'native'}"
        )
        typer.echo("Press Ctrl+C to stop.\n")

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        main_task = asyncio.create_task(relay_obj.run())
        shutdown_triggered = False

        def _on_sigint() -> None:
            nonlocal shutdown_triggered
            shutdown_triggered = True
            main_task.cancel()

        import signal

        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        try:
            await main_task
        except asyncio.CancelledError:
            pass
        finally:
            if shutdown_triggered:
                typer.echo("\nStopping relay...")
            loop.remove_signal_handler(signal.SIGINT)
            await relay_obj.teardown()
            if shutdown_triggered:
                typer.echo("Disconnected.")

    try:
        asyncio.run(_run())
    except RelayError as exc:
        typer.echo(f"\n✗ {exc}", err=True)
        raise typer.Exit(1)
