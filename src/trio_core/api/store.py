"""SQLite event store for Trio Console."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite


class EventStore:
    """Async SQLite event store for camera events and frames."""

    def __init__(
        self,
        db_path: str = "data/trio_console.db",
        frames_dir: str = "data/frames",
    ) -> None:
        self.db_path = db_path
        self.frames_dir = frames_dir
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Create tables and directories if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                camera_name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                frame_path TEXT,
                alert_triggered INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, timestamp);

            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                watch_condition TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            );
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ── Events ───────────────────────────────────────────────────────────────

    async def insert(self, event: dict) -> str:
        """Insert an event and return its ID."""
        assert self._db is not None
        event_id = event.get("id") or f"evt_{uuid.uuid4().hex[:12]}"
        ts = event.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        camera_id = event.get("camera_id", "")
        camera_name = event.get("camera_name", "")
        description = event.get("description", "")
        frame_path = event.get("frame_path")
        alert_triggered = 1 if event.get("alert_triggered") else 0
        metadata = json.dumps(event.get("metadata", {}))

        await self._db.execute(
            """
            INSERT INTO events (id, timestamp, camera_id, camera_name, description,
                                frame_path, alert_triggered, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, ts, camera_id, camera_name, description, frame_path, alert_triggered, metadata),
        )
        await self._db.commit()
        return event_id

    async def list_events(
        self,
        camera_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        alert_only: bool = False,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List events with filters. Returns {events, total}."""
        assert self._db is not None
        where_clauses: list[str] = []
        params: list[str | int] = []

        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end)
        if alert_only:
            where_clauses.append("alert_triggered = 1")
        if q:
            where_clauses.append("description LIKE ?")
            params.append(f"%{q}%")

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total count
        cursor = await self._db.execute(f"SELECT COUNT(*) FROM events {where}", params)
        row = await cursor.fetchone()
        total = row[0] if row else 0

        # Paginated results
        cursor = await self._db.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        rows = await cursor.fetchall()
        events = [_row_to_event(r) for r in rows]

        return {"events": events, "total": total}

    async def get_event(self, event_id: str) -> dict | None:
        """Get a single event by ID."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = await cursor.fetchone()
        return _row_to_event(row) if row else None

    async def summary(
        self,
        camera_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        """Aggregate event statistics."""
        assert self._db is not None
        where_clauses: list[str] = []
        params: list[str] = []

        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total events and alerts
        cursor = await self._db.execute(
            f"SELECT COUNT(*), SUM(alert_triggered) FROM events {where}", params
        )
        row = await cursor.fetchone()
        total_events = row[0] if row else 0
        total_alerts = int(row[1] or 0) if row else 0

        # By hour
        cursor = await self._db.execute(
            f"""
            SELECT SUBSTR(timestamp, 1, 13) AS hour, COUNT(*) AS cnt
            FROM events {where}
            GROUP BY hour ORDER BY hour
            """,
            params,
        )
        by_hour = {r[0]: r[1] for r in await cursor.fetchall()}

        # By camera
        cursor = await self._db.execute(
            f"SELECT camera_id, COUNT(*) FROM events {where} GROUP BY camera_id",
            params,
        )
        by_camera = {r[0]: r[1] for r in await cursor.fetchall()}

        return {
            "total_events": total_events,
            "total_alerts": total_alerts,
            "by_hour": by_hour,
            "by_camera": by_camera,
        }

    # ── Frames ───────────────────────────────────────────────────────────────

    async def save_frame(self, event_id: str, camera_id: str, frame_bytes: bytes) -> str:
        """Save a JPEG frame and return the relative path."""
        assert self._db is not None
        cam_dir = os.path.join(self.frames_dir, camera_id)
        os.makedirs(cam_dir, exist_ok=True)

        filename = f"{event_id}.jpg"
        rel_path = os.path.join(camera_id, filename)
        abs_path = os.path.join(self.frames_dir, rel_path)

        with open(abs_path, "wb") as f:
            f.write(frame_bytes)

        # Update event with frame path
        await self._db.execute(
            "UPDATE events SET frame_path = ? WHERE id = ?", (rel_path, event_id)
        )
        await self._db.commit()
        return rel_path

    async def get_frame_path(self, event_id: str) -> str | None:
        """Return absolute path to frame file, or None."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT frame_path FROM events WHERE id = ?", (event_id,)
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return None
        return os.path.abspath(os.path.join(self.frames_dir, row[0]))

    # ── Cameras ──────────────────────────────────────────────────────────────

    async def list_cameras(self) -> list[dict]:
        """List all cameras."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM cameras ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_camera(r) for r in rows]

    async def create_camera(self, camera: dict) -> str:
        """Create a camera and return its ID."""
        assert self._db is not None
        cam_id = camera.get("id") or f"cam_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat() + "Z"
        metadata = json.dumps(camera.get("metadata", {}))

        await self._db.execute(
            """
            INSERT INTO cameras (id, name, source_url, watch_condition, enabled, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cam_id,
                camera.get("name", ""),
                camera.get("source_url", ""),
                camera.get("watch_condition", ""),
                1 if camera.get("enabled", True) else 0,
                now,
                metadata,
            ),
        )
        await self._db.commit()
        return cam_id

    async def delete_camera(self, cam_id: str) -> bool:
        """Delete a camera. Returns True if it existed."""
        assert self._db is not None
        cursor = await self._db.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
        await self._db.commit()
        return cursor.rowcount > 0


def _row_to_event(row: aiosqlite.Row) -> dict:
    """Convert a database row to an event dict."""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "camera_name": row["camera_name"],
        "description": row["description"],
        "frame_path": row["frame_path"],
        "alert_triggered": bool(row["alert_triggered"]),
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


def _row_to_camera(row: aiosqlite.Row) -> dict:
    """Convert a database row to a camera dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "source_url": row["source_url"],
        "watch_condition": row["watch_condition"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }
