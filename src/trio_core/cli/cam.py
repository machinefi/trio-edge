from __future__ import annotations

import typer

from trio_core.cli._shared import app


def _resolve_rtsp_url(
    rtsp: str | None, host: str | None, port: int | None, user: str | None, password: str
) -> str:
    """Helper to resolve RTSP URL via explicit config or interactive discovery."""
    import sys

    from trio_core._rtsp_proxy import ensure_rtsp_url
    from trio_core.onvif import discover_cameras, get_rtsp_uri

    if rtsp:
        return ensure_rtsp_url(rtsp)

    # Use defaults if not provided via CLI
    user = user or "admin"
    port = port or 8000

    is_interactive = sys.stdin.isatty()

    if host:
        resolved_port = port
        typer.echo("Probing camera...")
        for camera in discover_cameras(timeout=2):
            if camera.ip == host:
                resolved_port = camera.port
                if camera.onvif_url:
                    typer.echo(f"Detected ONVIF: {camera.onvif_url}")
                break

        if not password:
            if not is_interactive:
                typer.echo("Password required for ONVIF. Please provide it via --password")
                raise typer.Exit(1)
            password = typer.prompt(f"Password for ONVIF user '{user}'", hide_input=True)

        rtsp_url = get_rtsp_uri(host, resolved_port, user, password)
        if not rtsp_url:
            typer.echo(
                "Failed to get RTSP URI. Credentials may be incorrect or camera unsupported."
            )
            raise typer.Exit(1)
        typer.echo(f"Using RTSP: {rtsp_url}")
        return ensure_rtsp_url(rtsp_url)

    # Interactive camera selection (no host or rtsp provided)
    typer.echo("Searching for ONVIF cameras on your network...\n")
    cameras = discover_cameras(timeout=3)

    if not cameras:
        typer.echo("No ONVIF cameras found. Try specifying --host <IP> -p <pass>")
        raise typer.Exit(1)

    c = cameras[0]
    if len(cameras) > 1:
        if not is_interactive:
            typer.echo(
                f"Multiple cameras ({len(cameras)}) found. Please specify one using --host <IP>"
            )
            raise typer.Exit(1)

        typer.echo(f"Found {len(cameras)} camera(s):\n")
        for i, cam in enumerate(cameras):
            typer.echo(f"  [{i + 1}] {cam.name}  IP: {cam.ip}:{cam.port}")

        try:
            choice_str = typer.prompt("\nSelect a camera", default="1")
            choice = int(choice_str) - 1
            if choice < 0 or choice >= len(cameras):
                raise ValueError()
            c = cameras[choice]
        except (ValueError, TypeError):
            typer.echo("Invalid selection.")
            raise typer.Exit(1)
    else:
        typer.echo(f"Found camera: {c.name} at {c.ip}:{c.port}")

    if not password:
        if not is_interactive:
            typer.echo("Password required for ONVIF. Please provide it via --password")
            raise typer.Exit(1)
        password = typer.prompt(f"Password for ONVIF user '{user}'", hide_input=True)

    rtsp_url = get_rtsp_uri(c.ip, c.port, user, password)
    if not rtsp_url:
        typer.echo("Failed to get RTSP URI. Credentials may be incorrect or camera unsupported.")
        raise typer.Exit(1)

    return ensure_rtsp_url(rtsp_url)


@app.command()
def cam(
    host: str = typer.Option(None, "--host", "-h", help="Camera IP address (skip discovery)"),
    port: int = typer.Option(None, "--port", help="ONVIF port"),
    user: str = typer.Option(None, "--user", "-u", help="Camera username"),
    password: str = typer.Option("", "--password", "-p", help="Camera password"),
    rtsp: str = typer.Option(None, "--rtsp", help="Direct RTSP URL (skip discovery + ONVIF)"),
    watch: str = typer.Option(None, "--watch", "-w", help="Watch condition in natural language"),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    max_tokens: int = typer.Option(10, "--max-tokens", help="Max generation tokens"),
    resolution: int = typer.Option(240, "--resolution", help="Max resolution (lower=faster)"),
    no_sound: bool = typer.Option(False, "--no-sound", help="Disable audio alerts"),
    count: bool = typer.Option(
        False, "--count", "-c", help="Count people, cars, dogs, cats (cumulative)"
    ),
    digest: bool = typer.Option(
        False, "--digest", "-d", help="Smart event timeline with scene understanding"
    ),
    adapter: str = typer.Option(None, "--adapter", "-a", help="LoRA adapter directory path"),
):
    """IP camera monitor with ONVIF discovery and AI analysis.

    Auto-discovers ONVIF cameras on the LAN, connects via RTSP, and runs
    live VLM analysis with a GUI preview window. Works through Tailscale.

    Examples:
        trio cam                                               # interactive discovery & select
        trio cam --host 192.168.1.100 -p pass                  # known IP
        trio cam --rtsp "rtsp://admin:pass@IP:554/stream"      # direct RTSP
        trio cam -w "someone at the door" --host 192.168.1.100 -p pass
        trio cam --rtsp "rtsp://..." --count                   # count objects
    """
    import os
    import sys

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRIO_COMPRESS_ENABLED", "1")
    os.environ.setdefault("TRIO_COMPRESS_RATIO", "0.5")

    # Resolve RTSP URL using the unified helper
    rtsp_url = _resolve_rtsp_url(rtsp, host, port, user, password)

    if not model:
        model = "mlx-community/Qwen3.5-2B-MLX-4bit"

    # Structured modes need more tokens
    if count and max_tokens < 40:
        max_tokens = 40
    if digest and max_tokens < 30:
        max_tokens = 30

    # Launch webcam GUI with RTSP source
    n_frames = "3" if digest else "1"
    sys.argv = [
        "webcam_gui",
        "--source",
        rtsp_url,
        "--frames",
        n_frames,
        "--max-tokens",
        str(max_tokens),
        "--interval",
        "0",
        "--resolution",
        str(resolution),
        "--model",
        model,
    ]
    if watch:
        sys.argv += ["--watch", watch]
    if backend:
        sys.argv += ["--backend", backend]
    if no_sound:
        sys.argv += ["--no-sound"]
    if count:
        sys.argv += ["--count"]
    if digest:
        sys.argv += ["--digest"]
    if adapter:
        sys.argv += ["--adapter", adapter]

    try:
        from trio_core._webcam_gui import main

        main()
    except KeyboardInterrupt:
        pass
