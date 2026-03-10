"""Lightweight health + Prometheus metrics HTTP server for trio-claw.

Runs on a configurable port alongside the main WebSocket event loop.
Designed to be minimal — no FastAPI/Flask dependency, just asyncio HTTP.
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger("trio.claw.health")


class HealthServer:
    """Async HTTP server for /health and /metrics endpoints."""

    def __init__(self, node, port: int = 9090):
        """
        Args:
            node: ClawNode instance (for status())
            port: TCP port to bind on
        """
        self._node = node
        self._port = port
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Start the HTTP server."""
        self._server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", self._port,
        )
        logger.info("Health server listening on port %d", self._port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            # Read the request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            if not request_line:
                return
            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ")
            method = parts[0] if parts else ""
            path = parts[1] if len(parts) > 1 else "/"

            # Drain headers (we don't need them)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break

            if method != "GET":
                self._send_response(writer, 405, "text/plain", "Method Not Allowed")
            elif path == "/health":
                self._handle_health(writer)
            elif path == "/metrics":
                self._handle_metrics(writer)
            else:
                self._send_response(writer, 404, "text/plain", "Not Found")

        except (asyncio.TimeoutError, ConnectionResetError):
            pass
        except Exception:
            logger.debug("Health server connection error", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /health — JSON status."""
        status = self._node.status()
        body = json.dumps(status, indent=2)
        self._send_response(writer, 200, "application/json", body)

    def _handle_metrics(self, writer: asyncio.StreamWriter) -> None:
        """GET /metrics — Prometheus text format."""
        lines = []
        status = self._node.status()
        handler = self._node.handler
        metrics = handler._metrics if handler and hasattr(handler, "_metrics") else None

        # Connection gauge
        connected = 1 if status["status"] == "connected" else 0
        lines.append("# HELP trio_claw_connected Whether the node is connected to Gateway")
        lines.append("# TYPE trio_claw_connected gauge")
        lines.append(f"trio_claw_connected {connected}")

        # Reconnects counter
        reconnects = status.get("reconnects", 0)
        lines.append("# HELP trio_claw_reconnects_total Total number of reconnections")
        lines.append("# TYPE trio_claw_reconnects_total counter")
        lines.append(f"trio_claw_reconnects_total {reconnects}")

        # Watch checks counter
        watch_checks = metrics.watch_checks if metrics else 0
        lines.append("# HELP trio_claw_watch_checks_total Total watch check iterations")
        lines.append("# TYPE trio_claw_watch_checks_total counter")
        lines.append(f"trio_claw_watch_checks_total {watch_checks}")

        # Watch alerts counter
        watch_alerts = metrics.watch_alerts if metrics else 0
        lines.append("# HELP trio_claw_watch_alerts_total Total watch alerts triggered")
        lines.append("# TYPE trio_claw_watch_alerts_total counter")
        lines.append(f"trio_claw_watch_alerts_total {watch_alerts}")

        # Capture failures counter
        capture_failures = metrics.capture_failures if metrics else 0
        lines.append("# HELP trio_claw_capture_failures_total Total camera capture failures")
        lines.append("# TYPE trio_claw_capture_failures_total counter")
        lines.append(f"trio_claw_capture_failures_total {capture_failures}")

        # Watches active gauge
        lines.append("# HELP trio_claw_watches_active Number of active watch loops")
        lines.append("# TYPE trio_claw_watches_active gauge")
        lines.append(f"trio_claw_watches_active {status['watches_active']}")

        # Uptime gauge
        lines.append("# HELP trio_claw_uptime_seconds Seconds since node started")
        lines.append("# TYPE trio_claw_uptime_seconds gauge")
        lines.append(f"trio_claw_uptime_seconds {status['uptime_s']}")

        # VLM latency histogram (manual buckets)
        if metrics and metrics.vlm_latency_samples:
            samples = metrics.vlm_latency_samples
            buckets = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            lines.append("# HELP trio_claw_vlm_latency_seconds VLM inference latency")
            lines.append("# TYPE trio_claw_vlm_latency_seconds histogram")
            total = len(samples)
            sum_val = sum(samples)
            for b in buckets:
                count = sum(1 for s in samples if s <= b)
                lines.append(f'trio_claw_vlm_latency_seconds_bucket{{le="{b}"}} {count}')
            lines.append(f'trio_claw_vlm_latency_seconds_bucket{{le="+Inf"}} {total}')
            lines.append(f"trio_claw_vlm_latency_seconds_sum {sum_val:.6f}")
            lines.append(f"trio_claw_vlm_latency_seconds_count {total}")

        body = "\n".join(lines) + "\n"
        self._send_response(writer, 200, "text/plain; version=0.0.4; charset=utf-8", body)

    @staticmethod
    def _send_response(
        writer: asyncio.StreamWriter,
        status_code: int,
        content_type: str,
        body: str,
    ) -> None:
        """Write an HTTP/1.1 response."""
        status_text = {200: "OK", 404: "Not Found", 405: "Method Not Allowed"}.get(
            status_code, "Unknown"
        )
        encoded = body.encode("utf-8")
        header = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(encoded)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode("utf-8") + encoded)
