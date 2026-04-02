from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.cli import app
from trio_core.onvif import CameraInfo

runner = CliRunner()


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
    assert "Garage (192.168.1.40)" in result.output
    assert "RTSP:" not in result.output


def test_cam_discover_lists_cameras_and_exits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [CameraInfo(name="Doorbell", ip="192.168.1.41", port=8899)],
    )

    result = runner.invoke(app, ["cam", "--discover"])

    assert result.exit_code == 0
    assert "Found 1 camera(s)" in result.output
    assert "[0] Doorbell  IP: 192.168.1.41:8899" in result.output


def test_cam_auto_discovery_uses_shared_helpers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "trio_core.onvif.discover_cameras",
        lambda timeout=5: [CameraInfo(name="Porch", ip="192.168.1.42", port=8000)],
    )
    monkeypatch.setattr(
        "trio_core.onvif.get_rtsp_uri",
        lambda host, port, user, password, fallback=True: "rtsp://admin:secret@192.168.1.42:554/stream1",
    )
    monkeypatch.setattr("trio_core._rtsp_proxy.ensure_rtsp_url", lambda url: url)

    captured = {}
    webcam_gui = types.ModuleType("trio_core._webcam_gui")

    def fake_main():
        captured["argv"] = list(sys.argv)

    webcam_gui.main = fake_main
    monkeypatch.setitem(sys.modules, "trio_core._webcam_gui", webcam_gui)

    result = runner.invoke(app, ["cam", "--password", "secret"])

    assert result.exit_code == 0
    assert captured["argv"][0] == "webcam_gui"
    assert "rtsp://admin:secret@192.168.1.42:554/stream1" in captured["argv"]


def test_cam_known_host_probes_for_onvif_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "trio_core.onvif.probe_camera",
        lambda host, ports=None, timeout=3.0: CameraInfo(
            name="Known Host",
            ip=host,
            port=2020,
            onvif_url=f"http://{host}:2020/onvif/service",
        ),
    )

    captured = {}

    def fake_get_rtsp_uri(host, port, user, password, fallback=True):
        captured["host"] = host
        captured["port"] = port
        return "rtsp://admin:secret@192.168.1.42:554/stream1"

    monkeypatch.setattr("trio_core.onvif.get_rtsp_uri", fake_get_rtsp_uri)
    monkeypatch.setattr("trio_core._rtsp_proxy.ensure_rtsp_url", lambda url: url)

    webcam_gui = types.ModuleType("trio_core._webcam_gui")

    def fake_main():
        captured["argv"] = list(sys.argv)

    webcam_gui.main = fake_main
    monkeypatch.setitem(sys.modules, "trio_core._webcam_gui", webcam_gui)

    result = runner.invoke(app, ["cam", "--host", "192.168.1.42", "--password", "secret"])

    assert result.exit_code == 0
    assert captured["host"] == "192.168.1.42"
    assert captured["port"] == 2020
    assert "Detected ONVIF: http://192.168.1.42:2020/onvif/service" in result.output


def test_cli_no_longer_imports_onvif_example():
    cli_text = Path("src/trio_core/cli.py").read_text()

    assert "onvif_monitor" not in cli_text
