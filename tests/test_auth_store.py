from __future__ import annotations

import json
import stat
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.auth_store import AuthStore, CameraEntry


def test_add_get_remove_and_list_cameras(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.json")

    store.add_camera("zeta", "192.168.1.20", 8554, "viewer", "pw-z")
    store.add_camera("alpha", "192.168.1.10", 554, "admin", "pw-a")

    assert store.get_camera("alpha") == CameraEntry(
        name="alpha",
        host="192.168.1.10",
        port=554,
        user="admin",
        password="pw-a",
    )
    assert store.get_camera("missing") is None
    assert [camera.name for camera in store.list_cameras()] == ["alpha", "zeta"]

    assert store.remove_camera("alpha") is True
    assert store.get_camera("alpha") is None
    assert [camera.name for camera in store.list_cameras()] == ["zeta"]
    assert store.remove_camera("missing") is False


def test_add_camera_upserts_existing_name(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.json")

    store.add_camera("office", "192.168.1.20", 554, "admin", "first")
    store.add_camera("office", "cam.local", 744, "alice", "second")

    assert store.list_cameras() == [
        CameraEntry(
            name="office",
            host="cam.local",
            port=744,
            user="alice",
            password="second",
        )
    ]


@pytest.mark.parametrize("name", ["bad name", "bad!name", "a" * 65])
def test_add_camera_rejects_invalid_names(tmp_path: Path, name: str):
    store = AuthStore(tmp_path / "auth.json")

    with pytest.raises(ValueError, match=r"Camera name must match"):
        store.add_camera(name, "192.168.1.20", 554, "admin", "secret")


def test_write_creates_missing_directory_valid_json_and_secure_permissions(tmp_path: Path):
    path = tmp_path / "config" / "auth.json"
    store = AuthStore(path)

    store.add_camera("office", "192.168.1.50", 554, "admin", "secret")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "office": {
            "host": "192.168.1.50",
            "password": "secret",
            "port": 554,
            "user": "admin",
        }
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_corrupt_json_returns_empty_list_and_warns(tmp_path: Path):
    path = tmp_path / "auth.json"
    path.write_text("{invalid json", encoding="utf-8")
    store = AuthStore(path)

    with pytest.warns(UserWarning, match=r"is corrupt"):
        assert store.list_cameras() == []


def test_empty_file_returns_empty_results(tmp_path: Path):
    path = tmp_path / "auth.json"
    path.write_text("", encoding="utf-8")
    store = AuthStore(path)

    assert store.list_cameras() == []
    assert store.get_camera("missing") is None


def test_read_auto_fixes_unsafe_permissions_and_warns(tmp_path: Path):
    path = tmp_path / "secure" / "auth.json"
    path.parent.mkdir(mode=0o755)
    path.write_text(
        json.dumps(
            {
                "office": {
                    "host": "192.168.1.50",
                    "port": 554,
                    "user": "admin",
                    "password": "secret",
                }
            }
        ),
        encoding="utf-8",
    )
    path.parent.chmod(0o755)
    path.chmod(0o644)
    store = AuthStore(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert store.get_camera("office") == CameraEntry(
            name="office",
            host="192.168.1.50",
            port=554,
            user="admin",
            password="secret",
        )

    messages = [str(item.message) for item in caught]
    assert any("directory" in message and "unsafe permissions" in message for message in messages)
    assert any("file" in message and "unsafe permissions" in message for message in messages)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
