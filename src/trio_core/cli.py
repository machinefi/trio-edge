"""CLI for TrioCore."""

from __future__ import annotations

import json
import logging
import shutil

import typer
from trio_core.http_ingest_relay import HttpIngestRelay, RelayError

app = typer.Typer(
    name="trio",
    help="Local VLM inference engine — analyze images and video on your Mac",
    no_args_is_help=True,
)


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        import json as _json
        import time as _t

        entry = {
            "ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["error"] = self.formatException(record.exc_info)
        return _json.dumps(entry, ensure_ascii=False)


def _setup_logging(verbose: bool = False, json_logs: bool = False) -> None:
    import os

    level = logging.DEBUG if verbose else logging.WARNING

    if json_logs or os.environ.get("TRIO_LOG_JSON", "").lower() in ("1", "true"):
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    if not verbose:
        # Suppress noisy "PyTorch was not found" warning from transformers
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        # Suppress httpx INFO logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        # Suppress huggingface_hub "Fetching N files" progress bars
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _version_callback(value: bool):
    if value:
        from trio_core import __version__

        typer.echo(f"trio-core {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
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

    # Optional deps
    try:
        import cv2

        typer.echo(f"opencv:    ✓ {cv2.__version__} (webcam support)")
    except ImportError:
        typer.echo(
            "opencv:    - not installed (optional, for webcam: pip install 'trio-core[webcam]')"
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


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8100, "--port", "-p", help="Bind port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    json_logs: bool = typer.Option(
        False, "--json-logs", help="Structured JSON logging (or set TRIO_LOG_JSON=1)"
    ),
):
    """Start the inference server (YOLO + VLM)."""
    _setup_logging(verbose, json_logs=json_logs)
    from trio_core.api.inference_server import main as run_server

    uv_level = "debug" if verbose else "info"
    run_server(host=host, port=port, log_level=uv_level)


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
    """Analyze an image or video."""
    _setup_logging(verbose)
    from trio_core.config import EngineConfig
    from trio_core.engine import TrioCore

    config = EngineConfig()
    if model:
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
        typer.echo(
            f"  Run {i + 1}/{runs}: {result.metrics.latency_ms:.0f}ms, {result.metrics.tokens_per_sec:.1f} tok/s"
        )

    avg = sum(latencies) / len(latencies)
    typer.echo(f"\n{runs} runs, avg {avg:.0f}ms")


_SMOKE_VIDEOS = [
    {
        "name": "surveillance_rotate",
        "url": "https://assets.mixkit.co/videos/48922/48922-720.mp4",
        "prompt": "You are a security camera AI. Describe what you see. Is there any suspicious activity?",
        "label": "Surveillance camera (rotating)",
    },
    {
        "name": "thieves_indoor",
        "url": "https://assets.mixkit.co/videos/31372/31372-720.mp4",
        "prompt": "You are a security camera AI monitoring a room. Describe any people and their behavior. Is anything suspicious?",
        "label": "Indoor security (thieves)",
    },
    {
        "name": "intruder_house",
        "url": "https://assets.mixkit.co/videos/12830/12830-720.mp4",
        "prompt": "Security alert: Is there a person? Describe their appearance and whether they appear to be an intruder.",
        "label": "House intrusion",
    },
]


def _ensure_smoke_videos() -> dict[str, str]:
    """Download smoke test videos if not cached. Returns {name: path}."""
    import os
    import urllib.request

    cache_dir = os.path.expanduser("~/.cache/trio-core/smoke-videos")
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for v in _SMOKE_VIDEOS:
        path = os.path.join(cache_dir, f"{v['name']}.mp4")
        if not os.path.exists(path):
            typer.echo(f"  Downloading {v['label']}...", nl=False)
            urllib.request.urlretrieve(v["url"], path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            typer.echo(f" {size_mb:.1f}MB")
        paths[v["name"]] = path
    return paths


@app.command()
def smoke(
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: mlx, transformers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Run end-to-end smoke tests to verify everything works.

    Tests model loading, real surveillance video inference, streaming,
    and API endpoints. Downloads test videos on first run (~18MB total).
    """
    _setup_logging(verbose)
    import asyncio
    import time

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

    typer.echo(f"\ntrio smoke — {config.model}\n")

    # Phase 0: Download test videos
    video_paths = _ensure_smoke_videos()

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

    # Phase 2: Surveillance video tests (3 real scenarios)
    for v in _SMOKE_VIDEOS:
        path = video_paths[v["name"]]

        def _video_test(p=path, prompt=v["prompt"]):
            r = engine.analyze_video(p, prompt, max_tokens=128)
            assert len(r.text.strip()) > 0, "empty response"
            return (
                f"{r.metrics.frames_input}→{r.metrics.frames_after_dedup}f, "
                f"{r.metrics.tokens_per_sec:.0f} tok/s, {len(r.text)} chars"
            )

        _run(v["label"], _video_test)

    # Phase 3: Streaming with real video
    stream_path = video_paths["intruder_house"]

    def _streaming():
        chunks = []

        async def _run_stream():
            async for chunk in engine.stream_analyze(
                stream_path,
                "Describe the scene in detail. What do you see?",
                max_tokens=128,
            ):
                if chunk.get("text"):
                    chunks.append(chunk["text"])

        asyncio.run(_run_stream())
        text = "".join(chunks)
        assert len(text.strip()) > 0, "empty streaming response"
        return f"{len(chunks)} chunks, {len(text)} chars"

    _run("Streaming (video)", _streaming)

    # Phase 4: API endpoints (using FastAPI TestClient, no server needed)
    def _api():
        try:
            from starlette.testclient import TestClient
        except ImportError:
            return "skipped (httpx not installed)"

        from trio_core.api.inference_server import create_app

        app_instance = create_app()

        client = TestClient(app_instance, raise_server_exceptions=False)

        # Health
        r = client.get("/health")
        assert r.status_code == 200, f"/health → {r.status_code}"

        # Inference status
        r = client.get("/api/inference/status")
        assert r.status_code == 200, f"/api/inference/status → {r.status_code}"

        return "2 endpoints OK"

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


def _resolve_rtsp_url(
    rtsp: str | None, host: str | None, port: int, user: str, password: str
) -> str:
    """Helper to resolve RTSP URL via explicit config or interactive discovery."""
    import sys

    from trio_core._rtsp_proxy import ensure_rtsp_url
    from trio_core.onvif import discover_cameras, get_rtsp_uri

    if rtsp:
        return ensure_rtsp_url(rtsp)

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
    port: int = typer.Option(8000, "--port", help="ONVIF port (default: 8000)"),
    user: str = typer.Option("admin", "--user", "-u", help="Camera username"),
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


def _die_load_error(e: Exception, model: str) -> None:
    """Print a friendly error message for model loading failures and exit."""
    msg = str(e)
    typer.echo(f"\n✗ Failed to load model: {model}", err=True)
    if "does not appear to have" in msg or "404" in msg or "not found" in msg.lower():
        typer.echo("  Model not found on HuggingFace. Check the model ID.", err=True)
        typer.echo(
            "  Example: trio analyze --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit video.mp4",
            err=True,
        )
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


@app.command()
def claw(
    gateway: str = typer.Option(
        "ws://127.0.0.1:18789", "--gateway", "-g", help="OpenClaw Gateway WebSocket URL"
    ),
    pair: bool = typer.Option(False, "--pair", help="Pair with Gateway (first-time setup)"),
    name: str = typer.Option("trio-core", "--name", "-n", help="Display name for this node"),
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    camera: list[str] = typer.Option(
        [], "--camera", "-c", help="Camera source (RTSP URL or device index). Repeatable."
    ),
    adapter: str = typer.Option(None, "--adapter", "-a", help="LoRA adapter directory"),
    token: str = typer.Option(None, "--token", "-t", help="Gateway auth token (pre-shared secret)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Structured JSON logging"),
    health_port: int = typer.Option(
        0, "--health-port", help="Enable health/metrics HTTP server on this port (e.g. 9090)"
    ),
    config: str = typer.Option(
        None, "--config", help="YAML config file (alternative to CLI flags)"
    ),
):
    """Connect trio-core as an OpenClaw node (replaces TrioClaw).

    First-time setup:
        trio claw --pair --gateway ws://host:18789 --token <gateway-secret>

    Run as node:
        trio claw --gateway ws://host:18789 --camera "rtsp://admin:pass@ip/stream"

    Production (with health endpoint):
        trio claw --gateway ws://host:18789 --health-port 9090 --json-logs

    Config file (alternative to CLI flags):
        trio claw --config trio-claw.yaml

    The node connects to the OpenClaw Gateway via WebSocket and handles
    vision/camera commands directly — no intermediate Go binary needed.
    """
    import asyncio
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Load config file if provided (CLI flags override config values)
    cfg = {}
    if config:
        cfg = _load_claw_config(config)

    # Merge config with CLI defaults (CLI flags win)
    gateway = gateway if gateway != "ws://127.0.0.1:18789" else cfg.get("gateway", gateway)
    model = model or cfg.get("model")
    adapter = adapter or cfg.get("adapter")
    token = token or cfg.get("token")
    health_port = health_port or cfg.get("health_port", 0)
    json_logs = json_logs or cfg.get("log_format") == "json"
    verbose = verbose or cfg.get("verbose", False)
    if not camera and "cameras" in cfg:
        camera = cfg["cameras"]

    _setup_logging(verbose, json_logs=json_logs)
    logging.getLogger("trio.claw").setLevel(logging.DEBUG if verbose else logging.INFO)

    try:
        from trio_core.claw.commands import CommandHandler
        from trio_core.claw.node import ClawNode
    except ImportError as e:
        typer.echo(f"Missing dependency: {e}")
        typer.echo("Install with: pip install 'trio-core[claw]'")
        raise typer.Exit(1)

    if pair:
        # Pairing mode — no engine needed
        node = ClawNode(gateway_url=gateway)
        if token:
            node.token = token  # gateway auth token for initial connect
        try:
            asyncio.run(node.pair(display_name=name))
            typer.echo("Paired! Token saved to ~/.trio/claw_state.json")
        except Exception as e:
            typer.echo(f"Pairing failed: {e}", err=True)
            raise typer.Exit(1)
        return

    # Normal mode — load engine (optional), connect as node
    engine = None
    if model or not camera:
        # Load VLM engine if model specified or no camera-only mode
        try:
            from trio_core import EngineConfig, TrioCore

            config_kwargs = {}
            if model:
                config_kwargs["model"] = model
            if adapter:
                config_kwargs["adapter_path"] = adapter
            config_obj = EngineConfig(**config_kwargs)

            typer.echo(f"Loading model: {config_obj.model} ...")
            engine = TrioCore(config_obj)
            engine.load()

            health = engine.health()
            typer.echo(f"Backend: {health.get('backend', {}).get('backend', 'unknown')}")
            typer.echo(f"Device: {health.get('backend', {}).get('device', 'unknown')}")
        except Exception as e:
            typer.echo(f"Warning: VLM engine not loaded ({e}). Camera-only mode.", err=True)

    # Set up camera sources
    sources = list(camera) if camera else ["0"]
    handler = CommandHandler(engine=engine, camera_sources=sources)
    node = ClawNode(gateway_url=gateway, handler=handler)
    if token:
        node.token = token  # CLI --token overrides saved token

    typer.echo(f"Connecting to Gateway: {gateway}")
    typer.echo(f"Cameras: {sources}")
    if health_port:
        typer.echo(f"Health endpoint: http://0.0.0.0:{health_port}/health")
    typer.echo("Press Ctrl+C to stop.\n")

    async def _run_with_health():
        health_server = None
        if health_port:
            from trio_core.claw.health import HealthServer

            health_server = HealthServer(node, port=health_port)
            await health_server.start()
        try:
            await node.run()
        finally:
            if health_server:
                await health_server.stop()

    try:
        asyncio.run(_run_with_health())
    except KeyboardInterrupt:
        typer.echo("\nDisconnected.")


def _load_claw_config(path: str) -> dict:
    """Load a YAML config file for trio claw."""
    try:
        import yaml
    except ImportError:
        typer.echo("YAML config requires PyYAML: pip install pyyaml", err=True)
        raise typer.Exit(1)

    from pathlib import Path

    p = Path(path)
    if not p.exists():
        typer.echo(f"Config file not found: {path}", err=True)
        raise typer.Exit(1)

    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return data


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


@app.command(help="HTTP MPEG-TS relay to Trio Cloud.")
def relay(
    cloud: str = typer.Option(
        ..., "--cloud", help="Trio Cloud base URL (e.g. https://api.trio.ai)"
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
    port: int = typer.Option(8000, "--port", help="ONVIF port (default: 8000)"),
    user: str = typer.Option("admin", "--user", "-u", help="Camera username"),
    password: str = typer.Option("", "--password", "-p", help="Camera password"),
    rtsp: str = typer.Option(None, "--rtsp", help="Direct RTSP URL (bypasses source)"),
    discover: bool = typer.Option(
        False, "--discover", help="Interactively discover ONVIF cameras to use as source"
    ),
    token: str = typer.Option(
        None, "--token", "-t", help="Bearer token for Trio Cloud authentication"
    ),
    resolution: str = typer.Option(
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


if __name__ == "__main__":
    app()
