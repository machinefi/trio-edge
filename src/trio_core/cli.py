"""CLI for TrioCore."""

from __future__ import annotations

import json
import logging
import shutil

import typer

app = typer.Typer(
    name="trio",
    help="Local VLM inference engine — analyze images and video on your Mac",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _version_callback(value: bool):
    if value:
        from trio_core import __version__
        typer.echo(f"trio-core {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
):
    pass


@app.command()
def doctor():
    """Check that everything is ready to run."""
    from trio_core import __version__
    from trio_core.device import detect_device, recommend_model

    typer.echo(f"trio-core {__version__}")
    typer.echo()

    all_ok = True

    # Hardware
    info = detect_device()
    model = recommend_model(info)
    typer.echo(f"Hardware:  {info.device_name} ({info.memory_gb:.0f}GB, {info.accelerator})")
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
            typer.echo("torch:     ✗ not installed → pip install 'trio-core[transformers]'")
            all_ok = False

    # Core deps
    try:
        import cv2
        typer.echo(f"opencv:    ✓ {cv2.__version__}")
    except ImportError:
        typer.echo("opencv:    ✗ not installed")
        all_ok = False

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
        ver = result.stdout.split("\n")[0].split("version ")[-1].split(" ")[0] if result.stdout else "unknown"
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
        free_gb = usage.free / (1024 ** 3)
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
        typer.echo("           First run will download the model (~2-5GB for 4-bit, ~15GB for 7B fp16).")
        typer.echo("           This may take 5-20 minutes depending on your connection.")

    typer.echo()
    if all_ok:
        typer.echo("✓ Ready. Run: trio serve")
    else:
        typer.echo("✗ Some dependencies missing. See above.")
        raise typer.Exit(1)


@app.command()
def serve(
    model: str = typer.Option(None, "--model", "-m", help="HuggingFace model ID or local path"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Start the API server."""
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
    video: str = typer.Argument(..., help="Image or video file path"),
    prompt: str = typer.Option(
        "Describe what you see.",
        "--prompt", "-q",
        help="Question or instruction",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max generation tokens"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output with metrics"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Analyze an image or video."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
        config.model = model

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
        typer.echo(f"\n{m.latency_ms:.0f}ms | {m.tokens_per_sec:.1f} tok/s | {m.prompt_tokens}+{m.completion_tokens} tokens")


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


@app.command()
def bench(
    video: str = typer.Argument(..., help="Video file path"),
    prompt: str = typer.Option("Describe this video.", "--prompt", "-q"),
    model: str = typer.Option(None, "--model", "-m"),
    backend: str = typer.Option(None, "--backend", "-b"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Benchmark inference speed."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
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
        typer.echo(f"  Run {i + 1}/{runs}: {result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.1f} tok/s")

    avg = sum(latencies) / len(latencies)
    typer.echo(f"\n{runs} runs, avg {avg:.0f}ms")


@app.command()
def smoke(
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Run end-to-end smoke tests to verify everything works."""
    _setup_logging(verbose)
    import asyncio
    import time

    import numpy as np
    from PIL import Image, ImageDraw

    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
        config.model = model

    results: list[tuple[str, bool, str]] = []  # (name, passed, detail)

    def _run(name: str, fn):
        typer.echo(f"  {name}...", nl=False)
        t0 = time.monotonic()
        try:
            detail = fn()
            dt = time.monotonic() - t0
            results.append((name, True, f"{detail} ({dt:.1f}s)"))
            typer.echo(f" ✓ {detail} ({dt:.1f}s)")
        except Exception as e:
            dt = time.monotonic() - t0
            results.append((name, False, f"{e} ({dt:.1f}s)"))
            typer.echo(f" ✗ {e} ({dt:.1f}s)")

    # Generate synthetic test image: colored rectangles with shapes
    def _make_test_image() -> np.ndarray:
        img = Image.new("RGB", (224, 224), (135, 206, 235))  # sky blue
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 120, 80, 200], fill=(34, 139, 34))   # green box
        draw.rectangle([140, 100, 200, 200], fill=(220, 20, 60))  # red box
        draw.ellipse([80, 30, 150, 100], fill=(255, 215, 0))      # yellow circle
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    # Generate synthetic test video: 4 frames with slight variation
    def _make_test_video() -> np.ndarray:
        frames = []
        for i in range(4):
            img = Image.new("RGB", (224, 224), (135, 206, 235))
            draw = ImageDraw.Draw(img)
            offset = i * 15
            draw.rectangle([20 + offset, 120, 80 + offset, 200], fill=(34, 139, 34))
            draw.ellipse([80, 30, 150, 100], fill=(255, 215, 0))
            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(arr.transpose(2, 0, 1))
        return np.stack(frames)  # (4, 3, 224, 224)

    typer.echo(f"\ntrio smoke — {config.model}\n")

    # Phase 1: Load model
    engine = TrioCore(config, backend=backend)

    def _load():
        engine.load()
        be = engine._backend
        return f"{be.backend_name} on {be.device_info.device_name}"
    _run("Load model", _load)

    if not engine._loaded:
        typer.echo("\n✗ Cannot continue — model failed to load.")
        raise typer.Exit(1)

    # Phase 2: Single image inference
    test_image = _make_test_image()

    def _single_image():
        r = engine.analyze_video(test_image, "What shapes and colors do you see?", max_tokens=64)
        assert len(r.text.strip()) > 0, "empty response"
        return f"{len(r.text)} chars, {r.metrics.tokens_per_sec:.1f} tok/s"
    _run("Single image", _single_image)

    # Phase 3: Multi-frame (video) inference
    test_video = _make_test_video()

    def _video():
        r = engine.analyze_video(test_video, "Describe what is happening.", max_tokens=64)
        assert len(r.text.strip()) > 0, "empty response"
        return f"{len(r.text)} chars, {r.metrics.prompt_tokens} prompt tokens"
    _run("Video (4 frames)", _video)

    # Phase 4: Streaming
    def _streaming():
        chunks = []

        async def _run_stream():
            async for chunk in engine.stream_analyze(
                test_image, "What do you see?", max_tokens=32,
            ):
                if chunk.get("text"):
                    chunks.append(chunk["text"])

        asyncio.run(_run_stream())
        text = "".join(chunks)
        assert len(text.strip()) > 0, "empty streaming response"
        return f"{len(chunks)} chunks, {len(text)} chars"
    _run("Streaming", _streaming)

    # Phase 5: API endpoints (using FastAPI TestClient, no server needed)
    def _api():
        try:
            from starlette.testclient import TestClient
        except ImportError:
            return "skipped (httpx not installed)"

        from trio_core.api.server import create_app, _engine as _old
        import trio_core.api.server as srv

        app_instance = create_app(config, backend=backend)
        # Inject already-loaded engine instead of loading again
        srv._engine = engine

        client = TestClient(app_instance, raise_server_exceptions=False)

        # Health
        r = client.get("/health")
        assert r.status_code == 200, f"/health → {r.status_code}"

        # /healthz
        r = client.get("/healthz")
        assert r.status_code == 200, f"/healthz → {r.status_code}"

        # /v1/models
        r = client.get("/v1/models")
        assert r.status_code == 200, f"/v1/models → {r.status_code}"

        # /analyze-frame with base64 image
        import base64
        import io
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        r = client.post("/analyze-frame", json={
            "frame_b64": b64,
            "question": "What color is this?",
        })
        assert r.status_code == 200, f"/analyze-frame → {r.status_code}: {r.text}"

        return f"4 endpoints OK"
    _run("API endpoints", _api)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    typer.echo(f"\n{'─' * 50}")
    for name, ok, detail in results:
        mark = "✓" if ok else "✗"
        typer.echo(f"  {mark} {name}: {detail}")
    typer.echo(f"{'─' * 50}")

    if passed == total:
        typer.echo(f"\n✓ All {total} checks passed. trio-core is ready.")
    else:
        typer.echo(f"\n✗ {passed}/{total} checks passed.")
        raise typer.Exit(1)


def _die_load_error(e: Exception, model: str) -> None:
    """Print a friendly error message for model loading failures and exit."""
    msg = str(e)
    typer.echo(f"\n✗ Failed to load model: {model}", err=True)
    if "does not appear to have" in msg or "404" in msg or "not found" in msg.lower():
        typer.echo("  Model not found on HuggingFace. Check the model ID.", err=True)
        typer.echo("  Example: trio analyze --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit video.mp4", err=True)
    elif "out of memory" in msg.lower() or "oom" in msg.lower() or isinstance(e, MemoryError):
        typer.echo("  Not enough memory. Try a smaller model (e.g. 3B-4bit).", err=True)
    elif "connection" in msg.lower() or "timeout" in msg.lower() or "resolve" in msg.lower():
        typer.echo("  Network error — check your internet connection.", err=True)
    elif "no module" in msg.lower() or isinstance(e, ImportError):
        typer.echo(f"  Missing dependency: {msg}", err=True)
        typer.echo("  Run: trio doctor", err=True)
    else:
        typer.echo(f"  {msg}", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
