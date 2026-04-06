from __future__ import annotations

import typer

from trio_core.cli._shared import app


@app.command()
def webcam(
    source: str = typer.Option("0", "--source", "-s", help="Camera index, RTSP URL, or video file"),
    watch: str = typer.Option(
        "a person is holding something in their hand",
        "--watch",
        "-w",
        help="Watch condition in natural language",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    max_tokens: int = typer.Option(10, "--max-tokens", help="Max generation tokens"),
    resolution: int = typer.Option(240, "--resolution", help="Max resolution (lower=faster)"),
    no_sound: bool = typer.Option(False, "--no-sound", help="Disable audio alerts"),
    count: bool = typer.Option(
        False, "--count", "-c", help="Count people, cars, dogs, cats (cumulative)"
    ),
):
    """Live webcam/camera monitor with VLM analysis and alerts.

    Uses natural language to define what to monitor — no ML training needed.
    Auto-calibrates resolution for ~1s inference on any Mac.

    Examples:
        trio webcam                                          # Default: detect holding objects
        trio webcam -w "a person is waving"                  # Custom condition
        trio webcam -w "someone at the door" -s 1            # iPhone Continuity Camera
        trio webcam -w "package on doorstep" -s rtsp://...   # IP camera
        trio webcam --count                                  # Count objects
    """
    import os
    import sys

    # Suppress noisy logs in webcam mode
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Set compression env vars for speed
    os.environ.setdefault("TRIO_COMPRESS_ENABLED", "1")
    os.environ.setdefault("TRIO_COMPRESS_RATIO", "0.5")

    # Default to Qwen3.5-2B — best accuracy under compression for real-time use
    # (Qwen2.5-VL-3B loses -19% POPE under compression, avoid for webcam)
    if not model:
        model = "mlx-community/Qwen3.5-2B-MLX-4bit"

    # Count mode needs more tokens
    if count and max_tokens < 40:
        max_tokens = 40

    # Build args
    sys.argv = [
        "webcam_gui",
        "--source",
        source,
        "--watch",
        watch,
        "--frames",
        "1",
        "--max-tokens",
        str(max_tokens),
        "--interval",
        "0",
        "--resolution",
        str(resolution),
        "--model",
        model,
    ]
    if backend:
        sys.argv += ["--backend", backend]
    if no_sound:
        sys.argv += ["--no-sound"]
    if count:
        sys.argv += ["--count"]

    try:
        from trio_core._webcam_gui import main

        main()
    except KeyboardInterrupt:
        pass
