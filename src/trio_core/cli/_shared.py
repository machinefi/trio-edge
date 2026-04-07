"""Shared utilities for CLI commands."""

from __future__ import annotations

import logging

import typer

app = typer.Typer(
    name="trio",
    help="Local VLM inference engine for image and video analysis",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
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
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _version_callback(value: bool):
    if value:
        from trio_core import __version__

        typer.echo(f"trio-core {__version__}")
        raise typer.Exit()


def _require_gpu() -> tuple[str, str]:
    """Detect hardware and return (model, backend). Exit on CPU-only."""
    from trio_core.device import detect_device, recommend_model

    info = detect_device()
    model = recommend_model(info)

    if info.accelerator == "cpu":
        typer.echo("✗ No GPU detected. trio-core requires Apple Silicon or NVIDIA GPU.", err=True)
        typer.echo("  Run: trio doctor", err=True)
        raise typer.Exit(1)

    return model, info.backend


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
