from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def test_discover_command_uses_shared_onvif_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Garage",
                ip="192.168.1.40",
                port=8000,
            )
        ],
    )

    result = runner.invoke(app, ["discover", "--timeout", "2"])

    assert result.exit_code == 0
    assert "Garage (192.168.1.40:8000)" in result.output
    assert "RTSP:" not in result.output


def test_cam_interactive_camera_listing_error_on_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [
            CameraInfo(name="Cam1", ip="192.168.1.41", port=8899),
            CameraInfo(name="Cam2", ip="192.168.1.42", port=8899),
        ],
    )

    result = runner.invoke(app, ["cam"])

    assert result.exit_code == 1
    assert "Multiple cameras (2) found. Please specify one using --host <IP>" in result.output


def test_cam_auto_discovery_uses_shared_helpers(monkeypatch: pytest.MonkeyPatch):
    captured = _mock_common_runtime(monkeypatch)
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [CameraInfo(name="Porch", ip="192.168.1.42", port=8000)],
    )
    monkeypatch.setattr(
        "trio_core.onvif.get_rtsp_uri",
        lambda host, port, user, password, fallback=True: (
            "rtsp://admin:secret@192.168.1.42:554/stream1"
        ),
    )

    result = runner.invoke(app, ["cam", "--password", "secret"])

    assert result.exit_code == 0
    assert captured["config"].source == "rtsp://admin:secret@192.168.1.42:554/stream1"


def test_cam_known_host_probes_for_onvif_port(monkeypatch: pytest.MonkeyPatch):
    _mock_common_runtime(monkeypatch)
    rtsp_captured: dict[str, object] = {}

    def fake_get_rtsp_uri(host, port, user, password, fallback=True):
        rtsp_captured["host"] = host
        rtsp_captured["port"] = port
        return "rtsp://admin:secret@192.168.1.42:554/stream1"

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
    monkeypatch.setattr("trio_core.onvif.get_rtsp_uri", fake_get_rtsp_uri)

    result = runner.invoke(app, ["cam", "--host", "192.168.1.42", "--password", "secret"])

    assert result.exit_code == 0
    assert rtsp_captured["host"] == "192.168.1.42"
    assert rtsp_captured["port"] == 2020
    assert "Detected ONVIF: http://192.168.1.42:2020/onvif/service" in result.output


def test_cli_no_longer_imports_onvif_example():
    cli_text = Path("src/trio_core/cli/cam.py").read_text()

    assert "onvif_monitor" not in cli_text
