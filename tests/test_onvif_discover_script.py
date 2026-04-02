from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from trio_core.onvif import CameraInfo


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "onvif_discover.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("onvif_discover_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onvif_discover_script_prints_cameras(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Front Door",
                ip="192.168.1.10",
                port=8000,
                onvif_url="http://192.168.1.10:8000/onvif/device_service",
                rtsp_url="rtsp://192.168.1.10:554/h264Preview_01_main",
            )
        ],
    )

    exit_code = module.main([])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Found 1 camera(s)" in out
    assert "[1] Front Door" in out
    assert "IP: 192.168.1.10:8000" in out
    assert "ONVIF: http://192.168.1.10:8000/onvif/device_service" in out
    assert "RTSP:" not in out


def test_onvif_discover_script_prints_empty_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()
    monkeypatch.setattr(module, "discover_cameras", lambda timeout=5: [])

    exit_code = module.main([])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.strip() == "No ONVIF cameras found on the network."


def test_onvif_discover_script_emits_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Garage",
                ip="192.168.1.40",
                port=8899,
                onvif_url="http://192.168.1.40:8899/onvif/device_service",
                rtsp_url="rtsp://192.168.1.40:554/stream1",
            )
        ],
    )

    exit_code = module.main(["--json"])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert exit_code == 0
    assert payload == [
        {
            "name": "Garage",
            "ip": "192.168.1.40",
            "port": 8899,
            "onvif_url": "http://192.168.1.40:8899/onvif/device_service",
            "scopes": None,
            "rtsp_url": None,
        }
    ]


def test_onvif_discover_script_probes_known_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Porch",
                ip="192.168.1.42",
                port=2020,
                onvif_url="http://192.168.1.42:2020/onvif/service",
                rtsp_url="rtsp://192.168.1.42:554/h264Preview_01_main",
            )
        ],
    )

    exit_code = module.main([])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[1] Porch" in out
    assert "IP: 192.168.1.42:2020" in out
    assert "ONVIF: http://192.168.1.42:2020/onvif/service" in out
    assert "RTSP:" not in out


def test_onvif_discover_script_resolves_rtsp_via_getstreamuri(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "discover_cameras",
        lambda timeout=5: [
            CameraInfo(
                name="Garage",
                ip="192.168.1.40",
                port=2020,
                onvif_url="http://192.168.1.40:2020/onvif/service",
                rtsp_url="rtsp://192.168.1.40:554/guessed",
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "get_rtsp_uri",
        lambda host, port, user, password, fallback=False: "rtsp://admin:pw@192.168.1.40:554/stream1",
    )

    exit_code = module.main(["--user", "admin", "--password", "pw"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "RTSP: rtsp://admin:pw@192.168.1.40:554/stream1" in out


def test_onvif_discover_script_suppresses_discovery_noise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_script_module()

    def noisy_discovery(timeout=5):
        print("transient discovery noise")
        return []

    monkeypatch.setattr(module, "discover_cameras", noisy_discovery)

    exit_code = module.main(["--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == []
