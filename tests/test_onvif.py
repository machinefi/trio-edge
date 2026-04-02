from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.onvif import CameraInfo, discover_cameras, get_rtsp_uri
import trio_core.onvif as onvif


def test_discover_cameras_falls_back_when_wsdiscovery_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    fallback = [CameraInfo(name="Fallback", ip="192.168.1.20", port=80)]

    def raise_import_error(timeout: int) -> list[CameraInfo]:
        raise ImportError("WSDiscovery missing")

    monkeypatch.setattr(onvif, "_discover_cameras_wsdiscovery", raise_import_error)
    monkeypatch.setattr(onvif, "_discover_cameras_probe", lambda timeout: fallback)

    assert discover_cameras() == fallback


def test_discover_cameras_falls_back_when_wsdiscovery_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    fallback = [CameraInfo(name="Probe", ip="192.168.1.30", port=80)]

    def raise_runtime_error(timeout: int) -> list[CameraInfo]:
        raise RuntimeError("broken discovery")

    monkeypatch.setattr(onvif, "_discover_cameras_wsdiscovery", raise_runtime_error)
    monkeypatch.setattr(onvif, "_discover_cameras_probe", lambda timeout: fallback)

    assert discover_cameras() == fallback


def test_discover_cameras_keeps_empty_wsdiscovery_result(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(onvif, "_discover_cameras_wsdiscovery", lambda timeout: [])

    def fail_if_called(timeout: int) -> list[CameraInfo]:
        raise AssertionError("probe fallback should not run on empty result")

    monkeypatch.setattr(onvif, "_discover_cameras_probe", fail_if_called)

    assert discover_cameras() == []


def test_discover_cameras_wsdiscovery_normalizes_camera_info(
    monkeypatch: pytest.MonkeyPatch,
):
    wsdiscovery_package = types.ModuleType("wsdiscovery")
    discovery_module = types.ModuleType("wsdiscovery.discovery")

    class FakeScope:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    class FakeService:
        def getScopes(self):
            return [
                FakeScope("onvif://www.onvif.org/type/NetworkVideoTransmitter"),
                FakeScope("onvif://www.onvif.org/name/Front%20Door"),
            ]

        def getXAddrs(self):
            return ["http://192.168.1.10:8000/onvif/device_service"]

    class ThreadedWSDiscovery:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def searchServices(self, timeout: int):
            return [FakeService()]

    discovery_module.ThreadedWSDiscovery = ThreadedWSDiscovery
    wsdiscovery_package.discovery = discovery_module
    monkeypatch.setitem(sys.modules, "wsdiscovery", wsdiscovery_package)
    monkeypatch.setitem(sys.modules, "wsdiscovery.discovery", discovery_module)

    cameras = onvif._discover_cameras_wsdiscovery(timeout=2)

    assert cameras == [
        CameraInfo(
            name="Front Door",
            ip="192.168.1.10",
            port=8000,
            onvif_url="http://192.168.1.10:8000/onvif/device_service",
            scopes=[
                "onvif://www.onvif.org/type/NetworkVideoTransmitter",
                "onvif://www.onvif.org/name/Front%20Door",
            ],
            rtsp_url="rtsp://192.168.1.10:554/h264Preview_01_main",
        )
    ]


def test_discover_cameras_probe_deduplicates_by_ip(monkeypatch: pytest.MonkeyPatch):
    responses = [
        (
            (
                b"<d:XAddrs>http://192.168.1.11:9000/onvif/device_service</d:XAddrs> "
                b"onvif://www.onvif.org/name/Back%20Gate"
            ),
            ("192.168.1.11", 3702),
        ),
        (
            (
                b"<d:XAddrs>http://192.168.1.11:9000/onvif/device_service</d:XAddrs> "
                b"onvif://www.onvif.org/name/Duplicate"
            ),
            ("192.168.1.11", 3702),
        ),
    ]

    class FakeSocket:
        def __init__(self, family, socktype) -> None:
            self._responses = list(responses)

        def settimeout(self, timeout: int) -> None:
            self.timeout = timeout

        def sendto(self, payload: bytes, addr) -> None:
            self.sent = (payload, addr)

        def recvfrom(self, size: int):
            if self._responses:
                return self._responses.pop(0)
            raise TimeoutError

        def close(self) -> None:
            self.closed = True

    fake_socket_module = types.SimpleNamespace(
        AF_INET=1,
        SOCK_DGRAM=2,
        timeout=TimeoutError,
        getaddrinfo=lambda *args, **kwargs: [(None, None, None, None, ("239.255.255.250", 3702))],
        socket=FakeSocket,
    )
    fake_time_module = types.SimpleNamespace(time=lambda: 0.0)

    monkeypatch.setitem(sys.modules, "socket", fake_socket_module)
    monkeypatch.setitem(sys.modules, "time", fake_time_module)

    cameras = onvif._discover_cameras_probe(timeout=1)

    assert len(cameras) == 1
    assert cameras[0] == CameraInfo(
        name="Back Gate",
        ip="192.168.1.11",
        port=9000,
        onvif_url="http://192.168.1.11:9000/onvif/device_service",
        scopes=["onvif://www.onvif.org/name/Back%20Gate"],
        rtsp_url="rtsp://192.168.1.11:554/h264Preview_01_main",
    )
    assert "@" not in (cameras[0].rtsp_url or "")


def test_get_rtsp_uri_injects_caller_credentials(monkeypatch: pytest.MonkeyPatch):
    onvif_module = types.ModuleType("onvif")

    class Profile:
        Name = "Primary"
        token = "token-1"

    class MediaService:
        def GetProfiles(self):
            return [Profile()]

        def GetStreamUri(self, params):
            return types.SimpleNamespace(Uri="rtsp://192.168.1.15:554/Streaming/Channels/101")

    class ONVIFCamera:
        def __init__(self, host: str, port: int, user: str, password: str) -> None:
            self.host = host
            self.port = port
            self.user = user
            self.password = password

        def create_media_service(self):
            return MediaService()

    onvif_module.ONVIFCamera = ONVIFCamera
    monkeypatch.setitem(sys.modules, "onvif", onvif_module)

    rtsp = get_rtsp_uri("192.168.1.15", 8000, "admin", "s3cr3t")

    assert rtsp == "rtsp://admin:s3cr3t@192.168.1.15:554/Streaming/Channels/101"


def test_get_rtsp_uri_keeps_existing_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        onvif,
        "_get_onvif_rtsp_uri",
        lambda host, port, user, password: "rtsp://viewer:pw@192.168.1.16:554/stream1",
    )

    rtsp = get_rtsp_uri("192.168.1.16", 8000, "admin", "s3cr3t")

    assert rtsp == "rtsp://viewer:pw@192.168.1.16:554/stream1"


def test_get_rtsp_uri_uses_fallback_with_caller_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_runtime_error(host: str, port: int, user: str, password: str) -> str | None:
        raise RuntimeError("camera rejected login")

    monkeypatch.setattr(onvif, "_get_onvif_rtsp_uri", raise_runtime_error)

    rtsp = get_rtsp_uri("192.168.1.17", 8000, "alice", "p@ss")

    assert rtsp == "rtsp://alice:p%40ss@192.168.1.17:554/h264Preview_01_main"


def test_get_rtsp_uri_returns_none_without_fallback(monkeypatch: pytest.MonkeyPatch):
    def raise_runtime_error(host: str, port: int, user: str, password: str) -> str | None:
        raise RuntimeError("camera rejected login")

    monkeypatch.setattr(onvif, "_get_onvif_rtsp_uri", raise_runtime_error)

    assert get_rtsp_uri("192.168.1.18", 8000, "alice", "secret", fallback=False) is None
