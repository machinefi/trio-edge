from __future__ import annotations

import typer

from trio_core.cli._shared import _setup_logging, app

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
