from __future__ import annotations

import typer

from trio_core.cli._shared import _setup_logging, app


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
