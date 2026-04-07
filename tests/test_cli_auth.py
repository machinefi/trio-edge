from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.auth_store import AuthStore
from trio_core.cli import app
from trio_core.password_input import LITERAL_PASSWORD_WARNING

runner = CliRunner()


def _patch_auth_store(monkeypatch: pytest.MonkeyPatch, path: Path) -> AuthStore:
    monkeypatch.setattr("trio_core.cli.auth.AuthStore", lambda: AuthStore(path))
    return AuthStore(path)


def _combined_output(result: object) -> str:
    output = getattr(result, "output", "")
    try:
        return output + result.stderr
    except Exception:
        return output


def test_auth_add_adds_new_camera_from_password_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    password_file = tmp_path / "password.txt"
    password_file.write_text("file-secret\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "auth",
            "add",
            "front-door",
            "--host",
            "192.168.1.10",
            "--user",
            "viewer",
            "--password-file",
            str(password_file),
        ],
    )

    assert result.exit_code == 0
    assert "Added front-door" in result.output

    camera = store.get_camera("front-door")
    assert camera is not None
    assert camera.host == "192.168.1.10"
    assert camera.port == 554
    assert camera.user == "viewer"
    assert camera.password == "file-secret"


def test_auth_add_updates_existing_camera_from_password_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    store.add_camera("front-door", "192.168.1.10", 554, "viewer", "old-secret")
    monkeypatch.setenv("TRIO_AUTH_PASSWORD", "env-secret")

    result = runner.invoke(
        app,
        [
            "auth",
            "add",
            "front-door",
            "--host",
            "192.168.1.11",
            "--port",
            "8554",
            "--user",
            "admin2",
            "--password-env",
            "TRIO_AUTH_PASSWORD",
        ],
    )

    assert result.exit_code == 0
    assert "Updated front-door" in result.output

    camera = store.get_camera("front-door")
    assert camera is not None
    assert camera.host == "192.168.1.11"
    assert camera.port == 8554
    assert camera.user == "admin2"
    assert camera.password == "env-secret"


def test_auth_add_requires_host_option():
    result = runner.invoke(app, ["auth", "add", "front-door", "--password", "secret"])

    assert result.exit_code == 2
    assert "Missing option '--host'" in result.output


def test_auth_add_rejects_invalid_camera_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    password_file = tmp_path / "password.txt"
    password_file.write_text("file-secret\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "auth",
            "add",
            "bad name",
            "--host",
            "192.168.1.12",
            "--password-file",
            str(password_file),
        ],
    )

    assert result.exit_code == 1
    assert "Camera name must match [a-zA-Z0-9_-]+" in result.output


def test_auth_add_warns_on_literal_password(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")

    result = runner.invoke(
        app,
        [
            "auth",
            "add",
            "garage",
            "--host",
            "192.168.1.13",
            "--password",
            "literal-secret",
        ],
    )

    assert result.exit_code == 0
    assert "Added garage" in result.output
    assert LITERAL_PASSWORD_WARNING in _combined_output(result)

    camera = store.get_camera("garage")
    assert camera is not None
    assert camera.password == "literal-secret"


def test_auth_add_fails_without_password_in_non_interactive_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    monkeypatch.setattr(
        "trio_core.cli.auth.sys",
        types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: False)),
    )

    result = runner.invoke(app, ["auth", "add", "garage", "--host", "192.168.1.13"])

    assert result.exit_code == 1
    assert "Password required. Use --password, --password-file, or --password-env." in result.output


def test_auth_status_lists_cameras_and_success_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    store.add_camera("alpha", "192.168.1.20", 554, "admin", "secret-a")
    store.add_camera("beta", "192.168.1.21", 8554, "viewer", "secret-b")

    calls: list[tuple[str, int, str, str, bool]] = []

    def fake_get_rtsp_uri(
        host: str, port: int, user: str, password: str, fallback: bool = True
    ) -> str:
        calls.append((host, port, user, password, fallback))
        return f"rtsp://{host}:{port}/stream"

    monkeypatch.setattr("trio_core.cli.auth.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 0
    assert "Name" in result.output
    assert "Host" in result.output
    assert "User" in result.output
    assert "Status" in result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "✅ OK" in result.output
    assert "2 camera(s): 2 OK, 0 failed" in result.output
    assert sorted(calls) == [
        ("192.168.1.20", 554, "admin", "secret-a", True),
        ("192.168.1.21", 8554, "viewer", "secret-b", True),
    ]


def test_auth_status_reports_empty_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_auth_store(monkeypatch, tmp_path / "auth.json")

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 0
    assert "No cameras configured. Use `trio auth add`." in result.output


def test_auth_status_exits_with_failure_when_any_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    store.add_camera("alpha", "192.168.1.30", 554, "admin", "secret-a")
    store.add_camera("beta", "192.168.1.31", 554, "viewer", "secret-b")

    def fake_get_rtsp_uri(
        host: str, port: int, user: str, password: str, fallback: bool = True
    ) -> str:
        if host == "192.168.1.31":
            raise RuntimeError("boom")
        return f"rtsp://{host}:{port}/stream"

    monkeypatch.setattr("trio_core.cli.auth.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 1
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "✅ OK" in result.output
    assert "❌ Connection Failed" in result.output
    assert "2 camera(s): 1 OK, 1 failed" in result.output


def test_auth_remove_deletes_existing_camera(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    store = _patch_auth_store(monkeypatch, tmp_path / "auth.json")
    store.add_camera("garage", "192.168.1.40", 554, "admin", "secret")

    result = runner.invoke(app, ["auth", "remove", "garage"])

    assert result.exit_code == 0
    assert "Removed garage" in result.output
    assert store.get_camera("garage") is None


def test_auth_remove_reports_missing_camera(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_auth_store(monkeypatch, tmp_path / "auth.json")

    result = runner.invoke(app, ["auth", "remove", "missing"])

    assert result.exit_code == 1
    assert "Camera 'missing' not found." in result.output
