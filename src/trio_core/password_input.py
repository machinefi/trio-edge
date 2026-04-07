from __future__ import annotations

import os
from pathlib import Path

import typer

LITERAL_PASSWORD_WARNING = (
    "⚠ WARNING: Password provided on command line is visible via /proc. "
    "Use --password (prompt) or --password-file instead."
)


def resolve_password(
    *,
    password_flag: str | None,
    password_file: str | None,
    password_env: str | None,
    is_interactive: bool = True,
) -> str | None:
    if password_flag == "":
        if not is_interactive:
            raise ValueError(
                "Cannot prompt for a password in non-interactive mode. "
                "Use --password-file, --password-env, or provide a literal --password value."
            )
        return typer.prompt("Password", hide_input=True)

    if password_file is not None:
        if not password_file:
            raise ValueError("Password file path cannot be empty.")
        try:
            return Path(password_file).read_text(encoding="utf-8").rstrip()
        except OSError as exc:
            raise ValueError(f"Failed to read password file '{password_file}': {exc}") from exc

    if password_env is not None:
        if not password_env:
            raise ValueError("Password environment variable name cannot be empty.")
        try:
            return os.environ[password_env]
        except KeyError as exc:
            raise ValueError(f"Environment variable '{password_env}' is not set.") from exc

    if password_flag:
        typer.echo(LITERAL_PASSWORD_WARNING, err=True)
        return password_flag

    return None
