from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.auth_store import CameraEntry
from trio_core.cli import app
from trio_core.onvif import CameraInfo

runner = CliRunner()


@dataclass
class _FakeWebcamGUIConfig:
    source: str
    watch: str | None = None
    model: str | None = None
    backend: str | None = None
    max_tokens: int = 10
    interval: int = 0
    frames: int = 1
    resolution: int = 240
    no_sound: bool = False
    count: bool = False
    digest: bool = False
    adapter: str | None = None


def _install_gui_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    webcam_gui = types.ModuleType("trio_core._webcam_gui")
    webcam_gui.WebcamGUIConfig = _FakeWebcamGUIConfig

    def fake_main(*, config):
        captured["config"] = config

    webcam_gui.main = fake_main
    monkeypatch.setitem(sys.modules, "trio_core._webcam_gui", webcam_gui)
    return captured


def _mock_common_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr("trio_core.cli.cam._require_gpu", lambda: ("test-model", "test-backend"))
    monkeypatch.setattr("trio_core._rtsp_proxy.ensure_rtsp_url", lambda url: url)
    return _install_gui_module(monkeypatch)


def test_cam_camera_resolves_registry_entry_and_launches_gui(monkeypatch: pytest.MonkeyPatch):
    captured = _mock_common_runtime(monkeypatch)
    requested: dict[str, object] = {}

    class FakeStore:
        def get_camera(self, name: str):
            requested["name"] = name
            return CameraEntry(
                name="office",
                host="192.168.1.50",
                port=9000,
                user="viewer",
                password="secret",
            )

    monkeypatch.setattr("trio_core.cli.cam.AuthStore", lambda: FakeStore())
    monkeypatch.setattr("trio_core.onvif.discover_cameras", lambda timeout=5: [])

    def fake_get_rtsp_uri(host, port, user, password, fallback=True):
        requested["rtsp"] = (host, port, user, password, fallback)
        return f"rtsp://{user}:{password}@{host}:554/stream1"

    monkeypatch.setattr("trio_core.onvif.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["cam", "--camera", "office"])

    assert result.exit_code == 0
    assert requested["name"] == "office"
    assert requested["rtsp"] == ("192.168.1.50", 9000, "viewer", "secret", True)
    config = captured["config"]
    assert config.source == "rtsp://viewer:secret@192.168.1.50:554/stream1"
    assert config.model == "test-model"
    assert config.backend == "test-backend"


def test_cam_camera_unknown_name_shows_auth_add_hint(monkeypatch: pytest.MonkeyPatch):
    class FakeStore:
        def get_camera(self, name: str):
            return None

    monkeypatch.setattr("trio_core.cli.cam.AuthStore", lambda: FakeStore())

    result = runner.invoke(app, ["cam", "--camera", "office"])

    assert result.exit_code == 1
    assert "Camera 'office' not found in registry. Use `trio auth add`." in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--camera", "office", "--source", "rtsp://example/stream"],
            "--camera cannot be used with --source",
        ),
        (["--camera", "office", "--host", "192.168.1.42"], "--camera cannot be used with --host"),
    ],
)
def test_cam_camera_rejects_conflicting_flags(
    monkeypatch: pytest.MonkeyPatch, args: list[str], message: str
):
    class FakeStore:
        def get_camera(self, name: str):
            raise AssertionError("registry lookup should not run for conflicting flags")

    monkeypatch.setattr("trio_core.cli.cam.AuthStore", lambda: FakeStore())

    result = runner.invoke(app, ["cam", *args])

    assert result.exit_code == 1
    assert message in result.output


def test_cam_source_regression_bypasses_onvif_resolution(monkeypatch: pytest.MonkeyPatch):
    captured = _mock_common_runtime(monkeypatch)
    source = "rtsp://viewer:secret@192.168.1.60:554/stream1"
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: (_ for _ in ()).throw(AssertionError("discovery should not run")),
    )
    monkeypatch.setattr(
        "trio_core.onvif.get_rtsp_uri",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RTSP lookup should not run")),
    )

    result = runner.invoke(app, ["cam", "--source", source])

    assert result.exit_code == 0
    assert captured["config"].source == source


def test_cam_host_and_password_regression_resolves_rtsp(monkeypatch: pytest.MonkeyPatch):
    captured = _mock_common_runtime(monkeypatch)
    requested: dict[str, object] = {}
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Known Host",
                ip="192.168.1.42",
                port=2020,
                onvif_url="http://192.168.1.42:2020/onvif/service",
            )
        ],
    )

    def fake_get_rtsp_uri(host, port, user, password, fallback=True):
        requested["rtsp"] = (host, port, user, password, fallback)
        return "rtsp://admin:secret@192.168.1.42:554/stream1"

    monkeypatch.setattr("trio_core.onvif.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["cam", "--host", "192.168.1.42", "--password", "secret"])

    assert result.exit_code == 0
    assert requested["rtsp"] == ("192.168.1.42", 2020, "admin", "secret", True)
    assert captured["config"].source == "rtsp://admin:secret@192.168.1.42:554/stream1"
    assert "Detected ONVIF: http://192.168.1.42:2020/onvif/service" in result.output


def test_cam_discovery_regression_resolves_rtsp(monkeypatch: pytest.MonkeyPatch):
    captured = _mock_common_runtime(monkeypatch)
    requested: dict[str, object] = {}
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [CameraInfo(name="Porch", ip="192.168.1.42", port=8000)],
    )

    def fake_get_rtsp_uri(host, port, user, password, fallback=True):
        requested["rtsp"] = (host, port, user, password, fallback)
        return "rtsp://admin:secret@192.168.1.42:554/stream1"

    monkeypatch.setattr("trio_core.onvif.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["cam", "--password", "secret"])

    assert result.exit_code == 0
    assert requested["rtsp"] == ("192.168.1.42", 8000, "admin", "secret", True)
    assert captured["config"].source == "rtsp://admin:secret@192.168.1.42:554/stream1"
    assert "Found camera: Porch at 192.168.1.42:8000" in result.output
