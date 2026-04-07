from __future__ import annotations

import sys
import threading
from queue import SimpleQueue

import typer
from click.core import ParameterSource

from trio_core.auth_store import AuthStore
from trio_core.cli._shared import app
from trio_core.onvif import get_rtsp_uri
from trio_core.password_input import resolve_password

auth_app = typer.Typer(help="Manage RTSP camera credentials")
app.add_typer(auth_app, name="auth")


@auth_app.command(help="Add or update stored RTSP camera credentials.")
def add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Camera name to store in the auth registry"),
    host: str = typer.Option(..., "--host", help="Camera IP address or hostname"),
    port: int = typer.Option(554, "--port", help="RTSP port for the camera"),
    user: str = typer.Option("admin", "--user", help="Camera username"),
    password: str = typer.Option(
        "",
        "--password",
        help="Camera password (pass '' to prompt interactively; literal values warn)",
    ),
    password_file: str | None = typer.Option(
        None,
        "--password-file",
        help="Read the camera password from a file",
    ),
    password_env: str | None = typer.Option(
        None,
        "--password-env",
        help="Read the camera password from an environment variable",
    ),
) -> None:
    store = AuthStore()
    existed = store.get_camera(name) is not None
    is_interactive = sys.stdin.isatty()

    # Determine what to pass to resolve_password.
    # --password was explicitly provided with a value → literal
    # --password was explicitly provided as '' → trigger prompt
    # --password not provided at all → treat as None (no source)
    password_was_explicit = ctx.get_parameter_source("password") is not ParameterSource.DEFAULT

    try:
        if password_was_explicit and password == "":
            # User explicitly passed --password '' → prompt
            if not is_interactive:
                typer.echo(
                    "Cannot prompt for password in non-interactive mode. "
                    "Use --password <value>, --password-file, or --password-env."
                )
                raise typer.Exit(1)
            resolved_password = typer.prompt("Password", hide_input=True)
        else:
            password_flag_value: str | None = password if password_was_explicit else None
            resolved_password = resolve_password(
                password_flag=password_flag_value,
                password_file=password_file,
                password_env=password_env,
                is_interactive=is_interactive,
            )
            if resolved_password is None:
                if not is_interactive:
                    typer.echo(
                        "Password required. Use --password, --password-file, or --password-env."
                    )
                    raise typer.Exit(1)
                resolved_password = typer.prompt(f"Password for {name}", hide_input=True)

        store.add_camera(name, host, port, user, resolved_password)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"{'Updated' if existed else 'Added'} {name}")


@auth_app.command(help="Show configured RTSP camera credentials and probe status.")
def status() -> None:
    cameras = AuthStore().list_cameras()
    if not cameras:
        typer.echo("No cameras configured. Use `trio auth add`.")
        raise typer.Exit(0)

    rows: list[tuple[str, str, str, str]] = []
    ok_count = 0
    failed_count = 0

    for camera in cameras:
        status_text = _probe_camera_status(camera.host, camera.port, camera.user, camera.password)
        if status_text == "✅ OK":
            ok_count += 1
        else:
            failed_count += 1
        rows.append((camera.name, camera.host, camera.user, status_text))

    widths = [
        max(len("Name"), *(len(row[0]) for row in rows)),
        max(len("Host"), *(len(row[1]) for row in rows)),
        max(len("User"), *(len(row[2]) for row in rows)),
        max(len("Status"), *(len(row[3]) for row in rows)),
    ]

    typer.echo(
        f"{'Name':<{widths[0]}}  {'Host':<{widths[1]}}  {'User':<{widths[2]}}  {'Status':<{widths[3]}}"
    )
    for name, host, user, status_text in rows:
        typer.echo(
            f"{name:<{widths[0]}}  {host:<{widths[1]}}  {user:<{widths[2]}}  {status_text:<{widths[3]}}"
        )

    typer.echo(f"{len(cameras)} camera(s): {ok_count} OK, {failed_count} failed")
    raise typer.Exit(0 if failed_count == 0 else 1)


def _probe_camera_status(host: str, port: int, user: str, password: str) -> str:
    result_queue: SimpleQueue[str | None] = SimpleQueue()
    error_queue: SimpleQueue[BaseException] = SimpleQueue()

    def _probe() -> None:
        try:
            result_queue.put(get_rtsp_uri(host, port, user, password, fallback=True))
        except BaseException as exc:
            error_queue.put(exc)

    thread = threading.Thread(target=_probe, daemon=True)
    thread.start()
    thread.join(timeout=5)
    if thread.is_alive():
        return "❌ Timeout"
    if not error_queue.empty():
        return "❌ Connection Failed"

    rtsp_uri = result_queue.get() if not result_queue.empty() else None
    if rtsp_uri:
        return "✅ OK"
    return "❌ Auth Failed"


@auth_app.command(help="Remove stored RTSP camera credentials.")
def remove(
    name: str = typer.Argument(..., help="Camera name to remove from the auth registry"),
) -> None:
    if not AuthStore().remove_camera(name):
        typer.echo(f"Camera '{name}' not found.")
        raise typer.Exit(1)

    typer.echo(f"Removed {name}")
