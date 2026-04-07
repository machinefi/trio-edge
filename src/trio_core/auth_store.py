from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path("~/.config/trio/auth.json").expanduser()
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(slots=True)
class CameraEntry:
    name: str
    host: str
    port: int = 554
    user: str = "admin"
    password: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
        }


class AuthStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or _DEFAULT_PATH).expanduser()

    def _ensure_dir(self) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, mode=_DIR_MODE, exist_ok=True)
        _ensure_mode(directory, _DIR_MODE, "directory")

    def _read(self) -> dict[str, CameraEntry]:
        self._ensure_dir()
        if not self.path.exists():
            return {}

        _ensure_mode(self.path, _FILE_MODE, "file")

        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.warn(
                f"Unable to read auth store {self.path}: {exc}. Treating as empty.",
                stacklevel=2,
            )
            return {}

        if not raw.strip():
            return {}

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            warnings.warn(
                f"Auth store {self.path} is corrupt. Treating as empty.",
                stacklevel=2,
            )
            return {}

        if not isinstance(payload, dict):
            warnings.warn(
                f"Auth store {self.path} must contain a JSON object. Treating as empty.",
                stacklevel=2,
            )
            return {}

        entries: dict[str, CameraEntry] = {}
        for name, value in payload.items():
            entry = _parse_entry(name, value)
            if entry is not None:
                entries[name] = entry
        return entries

    def _write(self, data: dict[str, CameraEntry]) -> None:
        self._ensure_dir()

        payload = {
            name: entry.to_dict() for name, entry in sorted(data.items(), key=lambda item: item[0])
        }

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, _FILE_MODE)
            os.replace(temp_path, self.path)
            os.chmod(self.path, _FILE_MODE)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def add_camera(
        self,
        name: str,
        host: str,
        port: int = 554,
        user: str = "admin",
        password: str = "",
    ) -> None:
        _validate_name(name)
        cameras = self._read()
        cameras[name] = CameraEntry(
            name=name,
            host=str(host),
            port=int(port),
            user=str(user),
            password=str(password),
        )
        self._write(cameras)

    def get_camera(self, name: str) -> CameraEntry | None:
        return self._read().get(name)

    def remove_camera(self, name: str) -> bool:
        cameras = self._read()
        if name not in cameras:
            return False
        del cameras[name]
        self._write(cameras)
        return True

    def list_cameras(self) -> list[CameraEntry]:
        cameras = self._read()
        return [cameras[name] for name in sorted(cameras)]


def _validate_name(name: str) -> None:
    if len(name) > 64 or not _NAME_RE.fullmatch(name):
        raise ValueError("Camera name must match [a-zA-Z0-9_-]+ and be at most 64 characters long.")


def _parse_entry(name: object, value: object) -> CameraEntry | None:
    if not isinstance(name, str):
        warnings.warn("Skipping auth store entry with a non-string camera name.", stacklevel=2)
        return None

    try:
        _validate_name(name)
    except ValueError:
        warnings.warn(f"Skipping auth store entry with invalid camera name {name!r}.", stacklevel=2)
        return None

    if not isinstance(value, Mapping):
        warnings.warn(f"Skipping auth store entry {name!r}: expected an object.", stacklevel=2)
        return None

    host = value.get("host")
    port = value.get("port", 554)
    user = value.get("user", "admin")
    password = value.get("password", "")

    if not isinstance(host, str) or not host:
        warnings.warn(f"Skipping auth store entry {name!r}: invalid host.", stacklevel=2)
        return None
    if not isinstance(port, int):
        warnings.warn(f"Skipping auth store entry {name!r}: invalid port.", stacklevel=2)
        return None
    if not isinstance(user, str):
        warnings.warn(f"Skipping auth store entry {name!r}: invalid user.", stacklevel=2)
        return None
    if not isinstance(password, str):
        warnings.warn(f"Skipping auth store entry {name!r}: invalid password.", stacklevel=2)
        return None

    return CameraEntry(name=name, host=host, port=port, user=user, password=password)


def _ensure_mode(path: Path, expected_mode: int, label: str) -> None:
    current_mode = stat.S_IMODE(path.stat().st_mode)
    if current_mode == expected_mode:
        return

    warnings.warn(
        f"Auth store {label} {path} has unsafe permissions {oct(current_mode)}; fixing to "
        f"{oct(expected_mode)}.",
        stacklevel=2,
    )
    os.chmod(path, expected_mode)
