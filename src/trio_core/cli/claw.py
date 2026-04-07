from __future__ import annotations

import logging

import typer

from trio_core.cli._shared import _setup_logging, app


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
    """Connect as an OpenClaw node for distributed inference.

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
