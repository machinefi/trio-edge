from __future__ import annotations

import shutil

import typer

from trio_core.cli._shared import app


@app.command()
def doctor():
    """Verify dependencies, hardware, and model readiness."""
    from trio_core import __version__
    from trio_core.device import detect_device, recommend_model

    typer.echo(f"trio-core {__version__}")
    typer.echo()

    all_ok = True

    # Hardware
    info = detect_device()
    model = recommend_model(info)
    typer.echo(f"Hardware:  {info.device_name} ({info.memory_gb:.0f}GB, {info.accelerator})")
    if info.compute_units:
        typer.echo(f"GPU cores: {info.compute_units}")
    typer.echo(f"Backend:   {info.backend}")
    typer.echo(f"Model:     {model}")

    # Backend deps
    typer.echo()
    if info.backend == "mlx":
        try:
            import mlx.core as mx

            typer.echo(f"mlx:       ✓ {mx.__version__}")
        except ImportError:
            typer.echo("mlx:       ✗ not installed → pip install 'trio-core[mlx]'")
            all_ok = False

        try:
            import mlx_vlm

            typer.echo(f"mlx-vlm:   ✓ {mlx_vlm.__version__}")
        except ImportError:
            typer.echo("mlx-vlm:   - not installed (optional, T2 models only)")
    else:
        try:
            import torch

            cuda = torch.cuda.is_available()
            typer.echo(f"torch:     ✓ {torch.__version__} (CUDA={cuda})")
        except ImportError:
            typer.echo("torch:     ✗ not installed → pip install 'trio-core[cuda]'")
            all_ok = False

    # Optional deps
    try:
        import cv2

        typer.echo(f"opencv:    ✓ {cv2.__version__} (webcam support)")
    except ImportError:
        typer.echo(
            "opencv:    - not installed (optional, for webcam: pip install 'trio-core[mlx]' or 'trio-core[cuda]')"
        )

    try:
        import PIL
        import PIL.Image  # noqa: F401 — verify PIL.Image is importable

        typer.echo(f"pillow:    ✓ {PIL.__version__}")
    except ImportError:
        typer.echo("pillow:    ✗ not installed")
        all_ok = False

    try:
        import fastapi

        typer.echo(f"fastapi:   ✓ {fastapi.__version__}")
    except ImportError:
        typer.echo("fastapi:   ✗ not installed")
        all_ok = False

    # ffmpeg (needed for video)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        import subprocess

        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver = (
            result.stdout.split("\n")[0].split("version ")[-1].split(" ")[0]
            if result.stdout
            else "unknown"
        )
        typer.echo(f"ffmpeg:    ✓ {ver}")
    else:
        typer.echo("ffmpeg:    ✗ not found → brew install ffmpeg")
        all_ok = False

    # Disk space
    typer.echo()
    import os

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        usage = shutil.disk_usage(os.path.dirname(cache_dir))
        free_gb = usage.free / (1024**3)
        if free_gb < 5:
            typer.echo(f"Disk:      ✗ {free_gb:.1f}GB free — need ≥5GB for model downloads")
            all_ok = False
        else:
            typer.echo(f"Disk:      ✓ {free_gb:.1f}GB free")
    except OSError:
        typer.echo("Disk:      ? unable to check free space")

    # Model cache
    if os.path.isdir(cache_dir):
        models = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
        typer.echo(f"HF cache:  {cache_dir} ({len(models)} models)")
    else:
        typer.echo(f"HF cache:  {cache_dir} (empty)")
        typer.echo(
            "           First run will download the model (~2-5GB for 4-bit, ~15GB for 7B fp16)."
        )
        typer.echo("           This may take 5-20 minutes depending on your connection.")

    typer.echo()
    if all_ok:
        typer.echo("✓ Ready. Run: trio serve")
    else:
        typer.echo("✗ Some dependencies missing. See above.")
        raise typer.Exit(1)
