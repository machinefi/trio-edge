from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.cli import app

runner = CliRunner()


def _relay_module():
    sys.modules.pop("trio_core.http_ingest_relay", None)
    return importlib.import_module("trio_core.http_ingest_relay")


def test_relay_command_help_mentions_cloud_http_ingest():
    result = runner.invoke(app, ["relay", "--help"])

    assert result.exit_code == 0
    assert "--cloud" in result.output
    assert "--camera-id" in result.output
    assert "HTTP MPEG-TS" in result.output


def test_relay_invalid_resolution_returns_error():
    result = runner.invoke(
        app,
        ["relay", "--cloud", "https://api.trio.ai", "--resolution", "bad"],
    )

    assert result.exit_code == 1
    assert "Invalid resolution format" in result.output


def test_derive_camera_id_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay.uuid, "getnode", lambda: 0xAABBCCDDEEFF)

    cam_a = relay.derive_camera_id("rtsp://camera/stream")
    cam_b = relay.derive_camera_id("rtsp://camera/stream")
    cam_c = relay.derive_camera_id("video.mp4")

    assert cam_a == cam_b
    assert cam_a != cam_c
    assert len(cam_a) == 36


def test_build_ffmpeg_cmd_rtsp_uses_mpegts_copy(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "rtsp")

    with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://camera/stream"):
        cmd = relay.HttpIngestRelay(
            source="rtsp://camera/stream",
            cloud_url="https://api.trio.ai",
            bearer_token="token",
        )._build_ffmpeg_cmd()

    assert "-f" in cmd
    assert "mpegts" in cmd
    assert "copy" in cmd
    assert "h264_mp4toannexb" not in " ".join(cmd)


def test_build_ffmpeg_cmd_webcam_macos_uses_mpegts(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "webcam")

    with patch("platform.system", return_value="Darwin"):
        cmd = relay.HttpIngestRelay(
            source="0",
            cloud_url="https://api.trio.ai",
            bearer_token="token",
            framerate=30,
        )._build_ffmpeg_cmd()

    assert "avfoundation" in cmd
    assert "libx264" in cmd
    assert "mpegts" in cmd
