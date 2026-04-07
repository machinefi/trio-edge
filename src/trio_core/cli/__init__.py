"""CLI for TrioCore."""

from __future__ import annotations

import typer

import trio_core.cli.analyze  # noqa: F401
import trio_core.cli.auth  # noqa: F401
import trio_core.cli.bench  # noqa: F401
import trio_core.cli.cam  # noqa: F401
import trio_core.cli.claw  # noqa: F401
import trio_core.cli.discover  # noqa: F401
import trio_core.cli.doctor  # noqa: F401
import trio_core.cli.relay  # noqa: F401
import trio_core.cli.serve  # noqa: F401
from trio_core.cli._shared import _JSONFormatter, _version_callback, app  # noqa: F401


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
) -> None:
    pass


if __name__ == "__main__":
    app()
