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

            CREATE TABLE IF NOT EXISTS alert_rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                camera_id TEXT,
                condition TEXT NOT NULL,
                channels TEXT DEFAULT '[]',
                cooldown_s INTEGER DEFAULT 60,
                enabled INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alert_history (
                id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                triggered_at TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT DEFAULT 'sent'
            );
            CREATE INDEX IF NOT EXISTS idx_alert_history_rule ON alert_history(rule_id);
            CREATE INDEX IF NOT EXISTS idx_alert_history_ts ON alert_history(triggered_at);

            CREATE TABLE IF NOT EXISTS metrics (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                event_id TEXT,
                metadata_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_camera_type_ts ON metrics(camera_id, metric_type, timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(timestamp);
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
            # Word-boundary-ish matching: add space padding
            where_clauses.append("(' ' || LOWER(description) || ' ') LIKE ?")
            params.append(f"% {q.lower()} %")

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

    # ── Camera Snapshots ──────────────────────────────────────────────────────

    async def save_camera_snapshot(self, camera_id: str, jpeg_bytes: bytes) -> str:
        """Save a camera snapshot to disk and return the file path."""
        cam_dir = os.path.join(self.frames_dir, camera_id)
        os.makedirs(cam_dir, exist_ok=True)
        path = os.path.join(cam_dir, "latest.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return path

    async def get_camera_snapshot(self, camera_id: str) -> bytes | None:
        """Read the latest snapshot for a camera from disk, or None."""
        path = os.path.join(self.frames_dir, camera_id, "latest.jpg")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read()

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


    # ── Alert Rules ───────────────────────────────────────────────────────

    async def list_alert_rules(self) -> list[dict]:
        """List all alert rules."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM alert_rules ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_alert_rule(r) for r in rows]

    async def create_alert_rule(self, rule: dict) -> str:
        """Create an alert rule and return its ID."""
        assert self._db is not None
        rule_id = f"rule_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat() + "Z"
        channels = json.dumps(rule.get("channels", []))

        await self._db.execute(
            """
            INSERT INTO alert_rules (id, name, camera_id, condition, channels, cooldown_s, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule_id,
                rule.get("name", ""),
                rule.get("camera_id"),
                rule.get("condition", ""),
                channels,
                rule.get("cooldown_s", 60),
                1 if rule.get("enabled", True) else 0,
                now,
            ),
        )
        await self._db.commit()
        return rule_id

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule. Returns True if it existed."""
        assert self._db is not None
        cursor = await self._db.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def list_alert_history(
        self,
        rule_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List alert history with filters. Returns {items, total}."""
        assert self._db is not None
        where_clauses: list[str] = []
        params: list[str | int] = []

        if rule_id:
            where_clauses.append("rule_id = ?")
            params.append(rule_id)
        if start:
            where_clauses.append("triggered_at >= ?")
            params.append(start)
        if end:
            where_clauses.append("triggered_at <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cursor = await self._db.execute(f"SELECT COUNT(*) FROM alert_history {where}", params)
        row = await cursor.fetchone()
        total = row[0] if row else 0

        cursor = await self._db.execute(
            f"SELECT * FROM alert_history {where} ORDER BY triggered_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        rows = await cursor.fetchall()
        items = [_row_to_alert_history(r) for r in rows]

        return {"items": items, "total": total}

    async def insert_alert_history(self, entry: dict) -> str:
        """Insert an alert history entry and return its ID."""
        assert self._db is not None
        entry_id = entry.get("id") or f"ah_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat() + "Z"

        await self._db.execute(
            """
            INSERT INTO alert_history (id, rule_id, event_id, triggered_at, channel, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                entry.get("rule_id", ""),
                entry.get("event_id", ""),
                entry.get("triggered_at", now),
                entry.get("channel", ""),
                entry.get("status", "sent"),
            ),
        )
        await self._db.commit()
        return entry_id

    # ── Metrics ──────────────────────────────────────────────────────────

    async def insert_metric(self, metric: dict) -> str:
        """Insert a metric data point and return its ID."""
        assert self._db is not None
        metric_id = metric.get("id") or f"met_{uuid.uuid4().hex[:12]}"
        ts = metric.get("timestamp") or datetime.utcnow().isoformat() + "Z"

        await self._db.execute(
            """
            INSERT INTO metrics (id, timestamp, camera_id, metric_type, value,
                                 confidence, event_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric_id,
                ts,
                metric.get("camera_id", ""),
                metric.get("metric_type", ""),
                metric.get("value", 0.0),
                metric.get("confidence", 1.0),
                metric.get("event_id"),
                json.dumps(metric.get("metadata", {})),
            ),
        )
        await self._db.commit()
        return metric_id

    async def query_metrics(
        self,
        camera_id: str,
        metric_type: str,
        start: str | None = None,
        end: str | None = None,
        granularity: str = "hour",
    ) -> list[dict]:
        """Return time-bucketed metric aggregations."""
        assert self._db is not None

        substr_len = {"minute": 16, "hour": 13, "day": 10}.get(granularity, 13)

        where_clauses = ["camera_id = ?", "metric_type = ?"]
        params: list[str | int] = [camera_id, metric_type]
        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end)

        where = "WHERE " + " AND ".join(where_clauses)

        cursor = await self._db.execute(
            f"""
            SELECT SUBSTR(timestamp, 1, {substr_len}) AS bucket,
                   SUM(value) AS total_value,
                   COUNT(*) AS cnt
            FROM metrics {where}
            GROUP BY bucket ORDER BY bucket
            """,
            params,
        )
        return [
            {"timestamp": r[0], "value": r[1], "count": r[2]}
            for r in await cursor.fetchall()
        ]

    async def latest_metrics(self, camera_id: str) -> dict:
        """Return the latest value for each metric_type for a camera."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT metric_type, value, timestamp
            FROM metrics
            WHERE camera_id = ?
              AND timestamp = (
                  SELECT MAX(m2.timestamp)
                  FROM metrics m2
                  WHERE m2.camera_id = metrics.camera_id
                    AND m2.metric_type = metrics.metric_type
              )
            GROUP BY metric_type
            """,
            (camera_id,),
        )
        return {r[0]: r[1] for r in await cursor.fetchall()}

    async def metrics_summary(
        self,
        camera_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        """Return aggregated metrics summary for a camera."""
        assert self._db is not None
        where_clauses = ["camera_id = ?"]
        params: list[str] = [camera_id]
        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end)
        where = "WHERE " + " AND ".join(where_clauses)

        # Total in/out
        cursor = await self._db.execute(
            f"SELECT COALESCE(SUM(value), 0) FROM metrics {where} AND metric_type = 'people_in'",
            params,
        )
        total_in = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            f"SELECT COALESCE(SUM(value), 0) FROM metrics {where} AND metric_type = 'people_out'",
            params,
        )
        total_out = (await cursor.fetchone())[0]

        # Average queue length
        cursor = await self._db.execute(
            f"SELECT COALESCE(AVG(value), 0) FROM metrics {where} AND metric_type = 'queue_length'",
            params,
        )
        avg_queue = round((await cursor.fetchone())[0], 2)

        # Peak hour (by people_in)
        cursor = await self._db.execute(
            f"""
            SELECT SUBSTR(timestamp, 1, 13) AS hour, SUM(value) AS cnt
            FROM metrics {where} AND metric_type = 'people_in'
            GROUP BY hour ORDER BY cnt DESC LIMIT 1
            """,
            params,
        )
        peak_row = await cursor.fetchone()
        peak_hour = peak_row[0] if peak_row else None
        peak_count = peak_row[1] if peak_row else 0

        return {
            "total_in": total_in,
            "total_out": total_out,
            "net_occupancy": total_in - total_out,
            "avg_queue": avg_queue,
            "peak_hour": peak_hour,
            "peak_count": peak_count,
        }

    # ── Reports ──────────────────────────────────────────────────────────

    async def daily_report(self, date_str: str, camera_id: str | None = None) -> dict:
        """Generate a daily report with hourly breakdown and per-camera stats."""
        assert self._db is not None
        day_start = f"{date_str}T00:00:00"
        day_end = f"{date_str}T23:59:59"

        where_clauses = ["timestamp >= ?", "timestamp <= ?"]
        params: list[str] = [day_start, day_end]
        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        where = "WHERE " + " AND ".join(where_clauses)

        # Totals
        cursor = await self._db.execute(
            f"SELECT COUNT(*), SUM(alert_triggered) FROM events {where}", params
        )
        row = await cursor.fetchone()
        total_events = row[0] if row else 0
        total_alerts = int(row[1] or 0) if row else 0

        # Hourly breakdown
        cursor = await self._db.execute(
            f"""
            SELECT CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) AS hour,
                   COUNT(*) AS events,
                   SUM(alert_triggered) AS alerts
            FROM events {where}
            GROUP BY hour ORDER BY hour
            """,
            params,
        )
        hourly = [
            {"hour": r[0], "events": r[1], "alerts": int(r[2] or 0)}
            for r in await cursor.fetchall()
        ]

        # Per-camera stats
        cursor = await self._db.execute(
            f"""
            SELECT camera_id, camera_name, COUNT(*) AS events, SUM(alert_triggered) AS alerts
            FROM events {where}
            GROUP BY camera_id
            """,
            params,
        )
        cameras = [
            {
                "camera_id": r[0],
                "camera_name": r[1] or r[0],
                "events": r[2],
                "alerts": int(r[3] or 0),
            }
            for r in await cursor.fetchall()
        ]

        # Anomalies (alert events)
        cursor = await self._db.execute(
            f"""
            SELECT id, timestamp, camera_id, camera_name, description
            FROM events {where} AND alert_triggered = 1
            ORDER BY timestamp DESC
            """,
            params,
        )
        anomalies = [
            {
                "id": r[0],
                "timestamp": r[1],
                "camera_id": r[2],
                "camera_name": r[3] or r[2],
                "description": r[4] or "",
            }
            for r in await cursor.fetchall()
        ]

        return {
            "date": date_str,
            "total_events": total_events,
            "total_alerts": total_alerts,
            "hourly": hourly,
            "cameras": cameras,
            "anomalies": anomalies,
        }

    async def trend_report(
        self, from_date: str, to_date: str, camera_id: str | None = None
    ) -> dict:
        """Generate a trend report over a date range."""
        assert self._db is not None
        day_start = f"{from_date}T00:00:00"
        day_end = f"{to_date}T23:59:59"

        where_clauses = ["timestamp >= ?", "timestamp <= ?"]
        params: list[str] = [day_start, day_end]
        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        where = "WHERE " + " AND ".join(where_clauses)

        # Daily counts
        cursor = await self._db.execute(
            f"""
            SELECT SUBSTR(timestamp, 1, 10) AS day,
                   COUNT(*) AS events,
                   SUM(alert_triggered) AS alerts
            FROM events {where}
            GROUP BY day ORDER BY day
            """,
            params,
        )
        days = [
            {"date": r[0], "events": r[1], "alerts": int(r[2] or 0)}
            for r in await cursor.fetchall()
        ]

        total_events = sum(d["events"] for d in days)
        total_alerts = sum(d["alerts"] for d in days)

        return {
            "from_date": from_date,
            "to_date": to_date,
            "days": days,
            "total_events": total_events,
            "total_alerts": total_alerts,
        }


def _row_to_alert_rule(row: aiosqlite.Row) -> dict:
    """Convert a database row to an alert rule dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "camera_id": row["camera_id"],
        "condition": row["condition"],
        "channels": json.loads(row["channels"] or "[]"),
        "cooldown_s": row["cooldown_s"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
    }


def _row_to_alert_history(row: aiosqlite.Row) -> dict:
    """Convert a database row to an alert history dict."""
    return {
        "id": row["id"],
        "rule_id": row["rule_id"],
        "event_id": row["event_id"],
        "triggered_at": row["triggered_at"],
        "channel": row["channel"],
        "status": row["status"],
    }


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
