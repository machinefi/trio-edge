from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.password_input import LITERAL_PASSWORD_WARNING, resolve_password


def test_resolve_password_literal_warns_and_returns_value(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_echo(message: str, err: bool = False):
        captured["message"] = message
        captured["err"] = err

    monkeypatch.setattr(typer, "echo", fake_echo)

    password = resolve_password(
        password_flag="super-secret",
        password_file=None,
        password_env=None,
    )

    assert password == "super-secret"
    assert captured == {"message": LITERAL_PASSWORD_WARNING, "err": True}


def test_resolve_password_reads_password_file_and_strips_trailing_whitespace(tmp_path: Path):
    password_file = tmp_path / "password.txt"
    password_file.write_text("from-file\n\n", encoding="utf-8")

    password = resolve_password(
        password_flag=None,
        password_file=str(password_file),
        password_env=None,
    )

    assert password == "from-file"


def test_resolve_password_reads_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIO_TEST_PASSWORD", "from-env")

    password = resolve_password(
        password_flag=None,
        password_file=None,
        password_env="TRIO_TEST_PASSWORD",
    )

    assert password == "from-env"


def test_resolve_password_raises_when_environment_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TRIO_MISSING_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="Environment variable 'TRIO_MISSING_PASSWORD' is not set"):
        resolve_password(
            password_flag=None,
            password_file=None,
            password_env="TRIO_MISSING_PASSWORD",
        )


def test_resolve_password_prefers_file_over_env_and_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    password_file = tmp_path / "password.txt"
    password_file.write_text("file-wins\n", encoding="utf-8")
    monkeypatch.setenv("TRIO_TEST_PASSWORD", "env-loses")

    password = resolve_password(
        password_flag="literal-loses",
        password_file=str(password_file),
        password_env="TRIO_TEST_PASSWORD",
    )

    assert password == "file-wins"


def test_resolve_password_prefers_env_over_literal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIO_TEST_PASSWORD", "env-wins")

    password = resolve_password(
        password_flag="literal-loses",
        password_file=None,
        password_env="TRIO_TEST_PASSWORD",
    )

    assert password == "env-wins"


def test_resolve_password_prompts_when_password_flag_requests_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_prompt(text: str, hide_input: bool = False):
        captured["text"] = text
        captured["hide_input"] = hide_input
        return "prompt-secret"

    monkeypatch.setattr(typer, "prompt", fake_prompt)

    password = resolve_password(
        password_flag="",
        password_file=None,
        password_env=None,
    )

    assert password == "prompt-secret"
    assert captured == {"text": "Password", "hide_input": True}


def test_resolve_password_rejects_prompt_in_non_interactive_mode():
    with pytest.raises(ValueError, match="Cannot prompt for a password in non-interactive mode"):
        resolve_password(
            password_flag="",
            password_file=None,
            password_env=None,
            is_interactive=False,
        )


def test_resolve_password_returns_none_when_no_sources_are_provided():
    password = resolve_password(
        password_flag=None,
        password_file=None,
        password_env=None,
    )

    assert password is None
