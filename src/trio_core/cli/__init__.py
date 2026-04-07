"""CLI for TrioCore."""

from __future__ import annotations

import typer

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


import trio_core.cli.analyze  # noqa: E402, F401
import trio_core.cli.bench  # noqa: E402, F401
import trio_core.cli.cam  # noqa: E402, F401
import trio_core.cli.claw  # noqa: E402, F401
import trio_core.cli.discover  # noqa: E402, F401
import trio_core.cli.doctor  # noqa: E402, F401
import trio_core.cli.relay  # noqa: E402, F401
import trio_core.cli.serve  # noqa: E402, F401

if __name__ == "__main__":
    app()
