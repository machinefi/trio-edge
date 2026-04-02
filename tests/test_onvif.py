from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import trio_core.onvif as onvif
from trio_core.onvif import CameraInfo, discover_cameras, get_rtsp_uri


def test_discover_cameras_uses_probe_path(monkeypatch: pytest.MonkeyPatch):
    cameras = [CameraInfo(name="Fallback", ip="192.168.1.21", port=2020)]
    monkeypatch.setattr(onvif, "_discover_cameras_probe", lambda timeout: cameras)

    assert discover_cameras(timeout=2) == cameras


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
        rtsp_url=None,
    )
    assert cameras[0].rtsp_url is None


def test_probe_response_parser_extracts_real_xaddr_and_scopes():
    response = """<?xml version="1.0" encoding="UTF-8"?>
    <SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
                       xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
                       xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
                       xmlns:tdn="http://www.onvif.org/ver10/network/wsdl">
      <SOAP-ENV:Body>
        <wsdd:ProbeMatches>
          <wsdd:ProbeMatch>
            <wsdd:Types>tdn:NetworkVideoTransmitter</wsdd:Types>
            <wsdd:Scopes>
              onvif://www.onvif.org/name/C120
              onvif://www.onvif.org/hardware/C120
              onvif://www.onvif.org/type/NetworkVideoTransmitter
            </wsdd:Scopes>
            <wsdd:XAddrs>http://192.168.6.215:2020/onvif/device_service</wsdd:XAddrs>
          </wsdd:ProbeMatch>
        </wsdd:ProbeMatches>
      </SOAP-ENV:Body>
    </SOAP-ENV:Envelope>"""

    camera = onvif._camera_info_from_probe_response(response, "192.168.6.215")

    assert camera == CameraInfo(
        name="C120",
        ip="192.168.6.215",
        port=2020,
        onvif_url="http://192.168.6.215:2020/onvif/device_service",
        scopes=[
            "onvif://www.onvif.org/name/C120",
            "onvif://www.onvif.org/hardware/C120",
            "onvif://www.onvif.org/type/NetworkVideoTransmitter",
        ],
        rtsp_url=None,
    )


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
