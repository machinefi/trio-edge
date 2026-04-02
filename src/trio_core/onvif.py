"""Shared ONVIF discovery and RTSP resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote, urlparse, urlunsplit

import httpx
import xml.etree.ElementTree as ET


_ONVIF_SCOPE = "onvif://www.onvif.org"
_COMMON_ONVIF_PORTS = (80, 8000, 8080, 8899, 2020)
_COMMON_ONVIF_PATHS = ("/onvif/device_service", "/onvif/service", "/device_service")
_DISCOVERY_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:trio-edge-discover</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>
  </e:Body>
</e:Envelope>"""
_CAPABILITIES_PROBE = """<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>
    <tds:GetCapabilities xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
      <tds:Category>All</tds:Category>
    </tds:GetCapabilities>
  </s:Body>
</s:Envelope>"""
_XADDR_RE = re.compile(r"https?://[^\s<\"]+")
_XADDR_ELEMENT_RE = re.compile(r"<(?:\w+:)?XAddr>\s*(https?://[^<]+)\s*</(?:\w+:)?XAddr>")


@dataclass(slots=True)
class CameraInfo:
    name: str
    ip: str
    port: int
    onvif_url: str | None = None
    scopes: list[str] | None = None
    rtsp_url: str | None = None


def discover_cameras(timeout: int = 5) -> list[CameraInfo]:
    """Discover ONVIF cameras on the local network."""
    try:
        cameras = _discover_cameras_wsdiscovery(timeout)
        if cameras:
            return cameras
        return _discover_cameras_probe(timeout)
    except ImportError:
        return _discover_cameras_probe(timeout)
    except Exception:
        return _discover_cameras_probe(timeout)


def probe_camera(host: str, ports: list[int] | None = None, timeout: float = 3.0) -> CameraInfo | None:
    """Probe a known host for an ONVIF device service on common ports."""
    candidates = _candidate_ports(ports)
    for port in candidates:
        for path in _COMMON_ONVIF_PATHS:
            url = f"http://{host}:{port}{path}"
            try:
                response = httpx.post(
                    url,
                    content=_CAPABILITIES_PROBE,
                    headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                    timeout=timeout,
                    follow_redirects=True,
                )
            except httpx.HTTPError:
                continue

            if response.status_code != 200 or not response.text:
                continue
            if "http://www.onvif.org/ver10/device/wsdl" not in response.text:
                continue

            onvif_url = _xaddr_from_capabilities_response(response.text) or url
            parsed = urlparse(onvif_url)
            if not parsed.hostname:
                continue

            return CameraInfo(
                name=f"Camera @ {host}",
                ip=host,
                port=parsed.port or port,
                onvif_url=onvif_url,
                scopes=None,
                rtsp_url=_build_discovery_rtsp_uri(host),
            )

    return None


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
        rtsp_uri = None

    if rtsp_uri:
        return _inject_rtsp_credentials(rtsp_uri, user, password)
    if fallback:
        return _build_fallback_rtsp_uri(host, user, password)
    return None


def _candidate_ports(ports: list[int] | None) -> list[int]:
    ordered = ports or list(_COMMON_ONVIF_PORTS)
    return list(dict.fromkeys(ordered))


def _discover_cameras_wsdiscovery(timeout: int) -> list[CameraInfo]:
    from wsdiscovery.discovery import ThreadedWSDiscovery

    wsd = ThreadedWSDiscovery()
    wsd.start()
    try:
        services = wsd.searchServices(timeout=timeout)
    finally:
        wsd.stop()

    cameras: list[CameraInfo] = []
    seen: set[str] = set()
    for svc in services:
        scopes = [str(s) for s in svc.getScopes()]
        if not any(_ONVIF_SCOPE in s or "NetworkVideoTransmitter" in s for s in scopes):
            continue

        xaddrs = svc.getXAddrs()
        if not xaddrs:
            continue

        onvif_url = str(xaddrs[0])
        parsed = urlparse(onvif_url)
        if not parsed.hostname or parsed.hostname in seen:
            continue

        seen.add(parsed.hostname)
        cameras.append(
            CameraInfo(
                name=_name_from_scopes(scopes, parsed.hostname),
                ip=parsed.hostname,
                port=parsed.port or 80,
                onvif_url=onvif_url,
                scopes=scopes,
                rtsp_url=_build_discovery_rtsp_uri(parsed.hostname),
            )
        )

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
        sock.sendto(_DISCOVERY_PROBE.encode(), addr[4])

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

    profile = profiles[0]
    uri = media.GetStreamUri(
        {
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            },
            "ProfileToken": profile.token,
        }
    )
    return getattr(uri, "Uri", None)


def _build_discovery_rtsp_uri(host: str, path: str = "/h264Preview_01_main") -> str:
    return f"rtsp://{host}:554{path}"


def _build_fallback_rtsp_uri(
    host: str,
    user: str,
    password: str,
    path: str = "/h264Preview_01_main",
) -> str:
    host_part = _format_host_for_netloc(host)
    userinfo = quote(user, safe="")
    if password or user:
        userinfo = f"{userinfo}:{quote(password, safe='')}@"
    return f"rtsp://{userinfo}{host_part}:554{path}"


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


def _xaddr_from_capabilities_response(response: str) -> str | None:
    match = _XADDR_ELEMENT_RE.search(response)
    if match:
        return match.group(1).strip()

    for candidate in _XADDR_RE.findall(response):
        parsed = urlparse(candidate)
        if parsed.hostname and "onvif" in parsed.path:
            return candidate
    return None


def _camera_info_from_probe_response(response: str, ip: str) -> CameraInfo | None:
    scopes = _scopes_from_probe_response(response)
    onvif_url = _xaddr_from_probe_response(response)
    port = urlparse(onvif_url).port if onvif_url else None
    return CameraInfo(
        name=_name_from_scopes(scopes, ip),
        ip=ip,
        port=port or 80,
        onvif_url=onvif_url,
        scopes=scopes or None,
        rtsp_url=_build_discovery_rtsp_uri(ip),
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
            candidate = scope.split("/name/")[-1].replace("%20", " ")
            if candidate:
                return candidate
        if "/hardware/" in scope and name.startswith("Camera @"):
            candidate = scope.split("/hardware/")[-1].replace("%20", " ")
            if candidate:
                name = f"{candidate} @ {ip}"
    return name
