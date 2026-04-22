"""Shared ONVIF discovery and RTSP resolution helpers."""

from __future__ import annotations

import logging
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse, urlunsplit

_ONVIF_SCOPE = "onvif://www.onvif.org"
_DISCOVERY_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>
  </e:Body>
</e:Envelope>"""
_XADDR_RE = re.compile(r"https?://[^\s<\"]+")


@dataclass(slots=True)
class CameraInfo:
    name: str
    ip: str
    port: int
    onvif_url: str | None = None
    scopes: list[str] | None = None
    rtsp_url: str | None = None


def discover_cameras(timeout: int = 5) -> list[CameraInfo]:
    """Discover ONVIF cameras on the local network.

    First tries WS-Discovery (UDP multicast). If that finds nothing, falls back
    to a direct TCP subnet scan probing common ONVIF ports.
    """
    try:
        cameras = _discover_cameras_probe(timeout)
        if cameras:
            return cameras
    except OSError:
        pass
    return _discover_cameras_subnet_scan(timeout)


def get_rtsp_uri(
    host: str,
    port: int,
    user: str,
    password: str,
    fallback: bool = True,
) -> str | None:
    """Resolve a camera RTSP URI via ONVIF, with optional fallback."""
    try:
        rtsp_uri = _get_onvif_rtsp_uri(host, port, user, password)
    except ImportError:
        rtsp_uri = None
    except Exception:
        logging.getLogger(__name__).exception("Failed to get RTSP URI from %s:%s", host, port)
        rtsp_uri = None

    if rtsp_uri:
        return _inject_rtsp_credentials(rtsp_uri, user, password)
    if fallback:
        return _build_fallback_rtsp_uri(host, user, password)
    return None


_ONVIF_PORTS = [2020, 8000, 80, 8080]


def _get_local_ip_from_hostname() -> str | None:
    import socket

    try:
        hostname = socket.gethostname()
        ip = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None


def _get_local_ip_from_udp() -> str | None:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _get_local_ip() -> str | None:
    """Return the local IP address or None."""
    return _get_local_ip_from_hostname() or _get_local_ip_from_udp()


def _get_local_subnet() -> str | None:
    """Return the local subnet prefix (e.g. '192.168.1') or None."""
    ip = _get_local_ip()
    if ip:
        return ".".join(ip.split(".")[:3])
    return None


def _probe_onvif_host(ip: str, timeout: float = 1.0) -> CameraInfo | None:
    """Check if a host has an ONVIF service on common ports via TCP connect."""
    import socket

    # First check if RTSP port 554 is open — real cameras have this
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        if s.connect_ex((ip, 554)) != 0:
            s.close()
            return None
        s.close()
    except Exception:
        return None

    for port in _ONVIF_PORTS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            result = s.connect_ex((ip, port))
            s.close()
            if result != 0:
                continue
            rtsp_path = _probe_rtsp_paths(ip, 554, "", "", timeout=timeout)
            if rtsp_path is None:
                continue
            return CameraInfo(
                name=f"Camera @ {ip}",
                ip=ip,
                port=port,
                onvif_url=f"http://{ip}:{port}/onvif/device_service",
                rtsp_url=f"rtsp://{ip}:554{rtsp_path}",
            )
        except Exception:
            continue
    return None


def _discover_cameras_subnet_scan(timeout: int) -> list[CameraInfo]:
    """Scan the local subnet for ONVIF cameras via direct TCP probe."""
    import concurrent.futures

    subnet = _get_local_subnet()
    if not subnet:
        return []

    cameras: list[CameraInfo] = []
    per_host_timeout = min(1.0, timeout / 10)
    ips = [f"{subnet}.{i}" for i in range(1, 255)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(_probe_onvif_host, ip, per_host_timeout): ip for ip in ips}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    if result is not None:
                        cameras.append(result)
                except Exception:
                    continue
        except TimeoutError:
            pass

    # Filter out the local machine
    local_ip = _get_local_ip()
    if local_ip:
        cameras = [c for c in cameras if c.ip != local_ip]

    return cameras


def _discover_cameras_probe(timeout: int) -> list[CameraInfo]:
    import socket
    import time

    addr = socket.getaddrinfo("239.255.255.250", 3702, socket.AF_INET, socket.SOCK_DGRAM)[0]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    cameras: list[CameraInfo] = []
    seen: set[str] = set()
    deadline = time.time() + timeout

    try:
        discovery_probe = _DISCOVERY_PROBE.format(message_id=f"uuid:{uuid.uuid4()}")
        try:
            sock.sendto(discovery_probe.encode(), addr[4])
        except OSError:
            sock.close()
            return []

        while time.time() < deadline:
            try:
                data, (ip, _) = sock.recvfrom(8192)
            except socket.timeout:
                break

            if ip in seen:
                continue
            seen.add(ip)

            response = data.decode(errors="replace")
            camera = _camera_info_from_probe_response(response, ip)
            if camera is not None:
                if not camera.rtsp_url:
                    camera.rtsp_url = get_rtsp_uri(camera.ip, camera.port, "", "")
                cameras.append(camera)
    finally:
        sock.close()

    return cameras


def _get_onvif_rtsp_uri(host: str, port: int, user: str, password: str) -> str | None:
    from onvif import ONVIFCamera

    cam = ONVIFCamera(host, port, user, password)
    media = cam.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        return None

    for profile in profiles:
        uri = media.GetStreamUri(
            {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": profile.token,
            }
        )
        uri_str = getattr(uri, "Uri", None) or getattr(uri, "uri", None)
        if uri_str:
            return uri_str
    return None


_FALLBACK_RTSP_PATHS = [
    "/stream1",
    "/h264Preview_01_main",
    "/live/ch00_0",
    "/cam/realmonitor?channel=1&subtype=0",
    "/ISAPI/Streaming/Channels/101",
]


def _build_fallback_rtsp_uri(
    host: str,
    user: str,
    password: str,
) -> str | None:
    probe_result = _probe_rtsp_paths(host, 554, user, password)
    if probe_result:
        host_part = _format_host_for_netloc(host)
        userinfo = quote(user, safe="")
        if password or user:
            userinfo = f"{userinfo}:{quote(password, safe='')}@"
        return f"rtsp://{userinfo}{host_part}:554{probe_result}"
    return None


def _get_onvif_rtsp_uri(host: str, port: int, user: str, password: str) -> str | None:
    from onvif import ONVIFCamera

    cam = ONVIFCamera(host, port, user, password)
    media = cam.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        return None

    for profile in profiles:
        uri = media.GetStreamUri(
            {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": profile.token,
            }
        )
        uri_str = getattr(uri, "Uri", None) or getattr(uri, "uri", None)
        if uri_str:
            return uri_str
    return None


_FALLBACK_RTSP_PATHS = [
    "/stream1",
    "/h264Preview_01_main",
    "/live/ch00_0",
    "/cam/realmonitor?channel=1&subtype=0",
    "/ISAPI/Streaming/Channels/101",
]


def _build_fallback_rtsp_uri(
    host: str,
    user: str,
    password: str,
) -> str | None:
    probe_result = _probe_rtsp_paths(host, 554, user, password)
    if probe_result:
        host_part = _format_host_for_netloc(host)
        userinfo = quote(user, safe="")
        if password or user:
            userinfo = f"{userinfo}:{quote(password, safe='')}@"
        return f"rtsp://{userinfo}{host_part}:554{probe_result}"
    return None


def _probe_rtsp_paths(
    host: str, port: int, user: str, password: str, timeout: float = 2.0
) -> str | None:
    import socket

    for path in _FALLBACK_RTSP_PATHS:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            req = (
                f"DESCRIBE rtsp://{host}:{port}{path} RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"Accept: application/sdp\r\n"
                f"\r\n"
            )
            sock.sendall(req.encode())
            resp = sock.recv(4096).decode(errors="replace")
            sock.close()
            if "200 OK" in resp or "302" in resp:
                return path
            if "401" in resp:
                return path
        except Exception:
            continue
    return None


def _inject_rtsp_credentials(url: str, user: str, password: str) -> str:
    parsed = urlparse(url)
    if parsed.username or not user:
        return url

    host = parsed.hostname
    if not host:
        return url

    host_part = _format_host_for_netloc(host)
    userinfo = quote(user, safe="")
    if password or user:
        userinfo = f"{userinfo}:{quote(password, safe='')}@"

    netloc = f"{userinfo}{host_part}"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _format_host_for_netloc(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _scopes_from_probe_response(response: str) -> list[str]:
    probe_match = _probe_match_from_xml(response)
    if probe_match is not None:
        scopes_text = _child_text(probe_match, "Scopes")
        if scopes_text:
            return [part for part in scopes_text.split() if part]

    scopes = []
    for part in response.split():
        if _ONVIF_SCOPE in part or "/name/" in part or "/hardware/" in part:
            scopes.append(part)
    return scopes


def _xaddr_from_probe_response(response: str) -> str | None:
    probe_match = _probe_match_from_xml(response)
    if probe_match is not None:
        xaddrs_text = _child_text(probe_match, "XAddrs")
        if xaddrs_text:
            for part in xaddrs_text.split():
                parsed = urlparse(part)
                if parsed.hostname:
                    return part

    match = _XADDR_RE.search(response)
    if not match:
        return None
    return match.group(0)


def _camera_info_from_probe_response(response: str, ip: str) -> CameraInfo:
    scopes = _scopes_from_probe_response(response)
    onvif_url = _xaddr_from_probe_response(response)
    parsed = urlparse(onvif_url) if onvif_url else None
    if parsed and parsed.port:
        port = parsed.port
    elif parsed and parsed.scheme == "https":
        port = 443
    elif parsed and parsed.scheme == "http":
        port = 80
    else:
        port = 80
    return CameraInfo(
        name=_name_from_scopes(scopes, ip),
        ip=ip,
        port=port,
        onvif_url=onvif_url,
        scopes=scopes or None,
        rtsp_url=None,
    )


def _probe_match_from_xml(response: str) -> ET.Element | None:
    try:
        root = ET.fromstring(response)
    except ET.ParseError:
        return None

    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] == "ProbeMatch":
            return elem
    return None


def _child_text(parent: ET.Element, local_name: str) -> str | None:
    for elem in parent.iter():
        if elem.tag.rsplit("}", 1)[-1] == local_name:
            text = (elem.text or "").strip()
            if text:
                return text
    return None


def _name_from_scopes(scopes: list[str] | None, ip: str) -> str:
    name = f"Camera @ {ip}"
    if not scopes:
        return name

    for scope in scopes:
        if "/name/" in scope:
            candidate = unquote(scope.split("/name/")[-1])
            if candidate:
                return candidate
        if "/hardware/" in scope and name.startswith("Camera @"):
            candidate = unquote(scope.split("/hardware/")[-1])
            if candidate:
                name = f"{candidate} @ {ip}"
    return name
