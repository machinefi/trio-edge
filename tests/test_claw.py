"""Tests for OpenClaw node integration — protocol, handshake, invoke dispatch."""

import asyncio
import json
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve

from trio_core.claw.protocol import (
    InvokeRequest,
    InvokeResult,
    connect_params,
    generate_id,
    make_event,
    make_req,
    make_res,
    pair_request_params,
)
from trio_core.claw.node import ClawNode, AuthError
from trio_core.claw.commands import (
    CommandHandler,
    NodeMetrics,
    WatchTask,
    _detect_triggered,
    CAPTURE_WARN_THRESHOLD,
    CAPTURE_STOP_THRESHOLD,
)


# =============================================================================
# Protocol unit tests
# =============================================================================

class TestProtocol:
    def test_make_req(self):
        frame = make_req("connect", {"role": "node"}, req_id="test1")
        assert frame["type"] == "req"
        assert frame["id"] == "test1"
        assert frame["method"] == "connect"
        assert frame["params"]["role"] == "node"

    def test_make_res(self):
        frame = make_res("test1", ok=True, payload={"status": "ok"})
        assert frame["type"] == "res"
        assert frame["id"] == "test1"
        assert frame["ok"] is True

    def test_make_event(self):
        frame = make_event("node.invoke.request", {"id": "inv1", "command": "test"})
        assert frame["type"] == "event"
        assert frame["event"] == "node.invoke.request"
        # payloadJSON is double-encoded
        payload = json.loads(frame["payloadJSON"])
        assert payload["id"] == "inv1"

    def test_generate_id(self):
        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 8
        assert id1 != id2  # statistically

    def test_connect_params(self):
        params = connect_params("test-node", token="tok123")
        assert params["client"]["id"] == "node-host"
        assert params["role"] == "node"
        assert params["auth"]["token"] == "tok123"
        assert "camera.snap" in params["commands"]

    def test_connect_params_no_token(self):
        params = connect_params("test-node")
        assert "auth" not in params

    def test_pair_request_params(self):
        params = pair_request_params("node1", "My Node")
        assert params["nodeId"] == "node1"
        assert params["displayName"] == "My Node"
        assert params["silent"] is False

    def test_invoke_request_from_payload(self):
        payload = json.dumps({
            "id": "inv1",
            "nodeId": "node1",
            "command": "vision.analyze",
            "params": {"question": "what do you see?"},
        })
        req = InvokeRequest.from_payload(payload)
        assert req.id == "inv1"
        assert req.command == "vision.analyze"
        assert req.params["question"] == "what do you see?"

    def test_invoke_request_double_encoded_params(self):
        """Params may be a JSON string (double-encoded)."""
        payload = json.dumps({
            "id": "inv2",
            "nodeId": "n",
            "command": "camera.snap",
            "params": json.dumps({"deviceId": "cam-0"}),
        })
        req = InvokeRequest.from_payload(payload)
        assert req.params["deviceId"] == "cam-0"

    def test_invoke_result_ok(self):
        result = InvokeResult(
            id="inv1", node_id="n", ok=True,
            payload={"answer": "a dog"},
        )
        frame = result.to_req_frame()
        assert frame["method"] == "node.invoke.result"
        params = frame["params"]
        assert params["ok"] is True
        assert json.loads(params["payloadJSON"])["answer"] == "a dog"

    def test_invoke_result_error(self):
        result = InvokeResult(
            id="inv1", node_id="n", ok=False,
            error_code="CAPTURE_FAILED", error_message="no camera",
        )
        frame = result.to_req_frame()
        params = frame["params"]
        assert params["ok"] is False
        assert params["error"]["code"] == "CAPTURE_FAILED"


# =============================================================================
# CommandHandler unit tests (no engine, no camera)
# =============================================================================

class TestCommandHandler:
    @pytest.fixture
    def handler(self):
        return CommandHandler(engine=None, camera_sources=["0", "rtsp://test"])

    @pytest.mark.asyncio
    async def test_camera_list(self, handler):
        req = InvokeRequest(id="1", node_id="n", command="camera.list")
        result = await handler.handle(req)
        assert result.ok
        assert len(result.payload["devices"]) == 2
        assert result.payload["devices"][0]["id"] == "cam-0"
        assert "RTSP" in result.payload["devices"][1]["name"]

    @pytest.mark.asyncio
    async def test_camera_list_masks_credentials(self):
        h = CommandHandler(engine=None, camera_sources=["rtsp://admin:pass@1.2.3.4/stream"])
        req = InvokeRequest(id="1", node_id="n", command="camera.list")
        result = await h.handle(req)
        assert "***" in result.payload["devices"][0]["name"]
        assert "pass" not in result.payload["devices"][0]["name"]

    @pytest.mark.asyncio
    async def test_vision_analyze_no_engine(self, handler):
        req = InvokeRequest(
            id="1", node_id="n", command="vision.analyze",
            params={"question": "test"},
        )
        result = await handler.handle(req)
        # Should fail with CAPTURE_FAILED (no camera) or ENGINE_NOT_LOADED
        assert not result.ok

    @pytest.mark.asyncio
    async def test_vision_analyze_no_question(self, handler):
        req = InvokeRequest(
            id="1", node_id="n", command="vision.analyze",
            params={},
        )
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_unknown_command(self, handler):
        req = InvokeRequest(id="1", node_id="n", command="foo.bar")
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_vision_status(self, handler):
        req = InvokeRequest(id="1", node_id="n", command="vision.status")
        result = await handler.handle(req)
        assert result.ok
        assert "watches" in result.payload

    def test_resolve_source(self, handler):
        assert handler._resolve_source("") == "0"  # default
        assert handler._resolve_source("cam-1") == "rtsp://test"
        assert handler._resolve_source("cam-99") == "cam-99"  # out of range
        assert handler._resolve_source("2") == "2"  # direct index


# =============================================================================
# NodeMetrics unit tests
# =============================================================================

class TestNodeMetrics:
    def test_initial_state(self):
        m = NodeMetrics()
        assert m.watch_checks == 0
        assert m.capture_failures == 0
        assert m.vlm_latency_samples == []

    def test_record_vlm_latency(self):
        m = NodeMetrics()
        m.record_vlm_latency(0.5)
        m.record_vlm_latency(1.2)
        assert len(m.vlm_latency_samples) == 2

    def test_latency_capped_at_1000(self):
        m = NodeMetrics()
        for i in range(1100):
            m.record_vlm_latency(float(i))
        assert len(m.vlm_latency_samples) == 1000
        # Should keep the latest 1000
        assert m.vlm_latency_samples[0] == 100.0


# =============================================================================
# Node integration test (mock Gateway)
# =============================================================================

class TestClawNode:
    @pytest.mark.asyncio
    async def test_handshake_and_invoke(self):
        """Full flow: connect → handshake → invoke → result."""
        results_received = []
        got_result = asyncio.Event()

        async def mock_gateway(ws):
            # 1. Send challenge
            await ws.send(json.dumps({
                "type": "event",
                "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "abc123"}),
            }))

            # 2. Receive connect request
            frame = json.loads(await ws.recv())
            assert frame["type"] == "req"
            assert frame["method"] == "connect"
            assert frame["params"]["auth"]["token"] == "test-token"

            # 3. Send hello-ok
            await ws.send(json.dumps({
                "type": "res",
                "id": frame["id"],
                "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # 4. Send invoke request (camera.list — works without camera)
            invoke_payload = {
                "id": "inv-test",
                "nodeId": "test-node",
                "command": "camera.list",
                "params": {},
            }
            await ws.send(json.dumps({
                "type": "event",
                "event": "node.invoke.request",
                "payloadJSON": json.dumps(invoke_payload),
            }))

            # 5. Receive invoke result
            result_frame = json.loads(await ws.recv())
            results_received.append(result_frame)
            got_result.set()

            # Keep connection open briefly so node doesn't reconnect
            await asyncio.sleep(0.5)

        async with serve(mock_gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]

            handler = CommandHandler(engine=None, camera_sources=["0"])
            node = ClawNode(
                gateway_url=f"ws://127.0.0.1:{port}",
                node_id="test-node",
                handler=handler,
            )
            node.token = "test-token"

            async def run_and_stop():
                task = asyncio.create_task(node.run())
                await asyncio.wait_for(got_result.wait(), timeout=5)
                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

            await run_and_stop()

        # Verify result
        assert len(results_received) == 1
        result = results_received[0]
        assert result["method"] == "node.invoke.result"
        params = result["params"]
        assert params["ok"] is True
        payload = json.loads(params["payloadJSON"])
        assert "devices" in payload

    @pytest.mark.asyncio
    async def test_ping_response(self):
        """Node responds to gateway ping with pong (res)."""
        pong_received = asyncio.Event()

        async def mock_gateway(ws):
            # Handshake
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": "{}",
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # Send ping request
            await ws.send(json.dumps({
                "type": "req", "id": "ping1", "method": "ping", "params": {},
            }))

            # Expect pong response
            resp = json.loads(await ws.recv())
            if resp.get("type") == "res" and resp.get("id") == "ping1":
                pong_received.set()

            await ws.close()

        async with serve(mock_gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            node = ClawNode(gateway_url=f"ws://127.0.0.1:{port}")
            node.token = "tok"

            task = asyncio.create_task(node.run())
            await asyncio.wait_for(pong_received.wait(), timeout=5)
            await node.stop()
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.TimeoutError, Exception):
                task.cancel()

        assert pong_received.is_set()

    @pytest.mark.asyncio
    async def test_auth_failure_no_retry(self):
        """Node exits immediately on auth rejection (no reconnect loop)."""
        connect_attempts = 0

        async def mock_gateway(ws):
            nonlocal connect_attempts
            connect_attempts += 1
            # Send challenge
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n1"}),
            }))
            # Receive connect
            frame = json.loads(await ws.recv())
            # Reject with auth error
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": False,
                "error": {"code": "AUTH_FAILED", "message": "Invalid token"},
            }))
            await ws.close()

        async with serve(mock_gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            node = ClawNode(gateway_url=f"ws://127.0.0.1:{port}")
            node.token = "bad-token"

            # run() should complete without retrying
            await asyncio.wait_for(node.run(), timeout=5)

        # Should have connected exactly once (no retry on auth failure)
        assert connect_attempts == 1

    @pytest.mark.asyncio
    async def test_status_method(self):
        """node.status() returns structured status dict."""
        handler = CommandHandler(engine=None, camera_sources=["0"])
        node = ClawNode(gateway_url="ws://localhost:1234", handler=handler)
        node.token = "tok"
        node._start_time = 100.0

        with patch("time.monotonic", return_value=110.0):
            s = node.status()

        assert s["status"] == "reconnecting"  # not connected
        assert s["uptime_s"] == 10.0
        assert s["watches_active"] == 0
        assert s["gateway"] == "ws://localhost:1234"

    @pytest.mark.asyncio
    async def test_graceful_shutdown_via_stop(self):
        """node.stop() cleanly terminates the run loop."""
        handshake_done = asyncio.Event()

        async def mock_gateway(ws):
            # Handshake
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n1"}),
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))
            handshake_done.set()
            # Keep connection open
            try:
                async for _ in ws:
                    pass
            except websockets.ConnectionClosed:
                pass

        async with serve(mock_gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            node = ClawNode(gateway_url=f"ws://127.0.0.1:{port}")
            node.token = "tok"

            task = asyncio.create_task(node.run())
            await asyncio.wait_for(handshake_done.wait(), timeout=5)

            # Stop should complete cleanly
            await node.stop()
            await asyncio.wait_for(task, timeout=3)

        # Task completed without error
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_reconnect_stops_watches(self):
        """On reconnect, stale watches from previous connection are cleared."""
        first_connect = True
        reconnected = asyncio.Event()

        async def mock_gateway(ws):
            nonlocal first_connect
            # Handshake
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n1"}),
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            if first_connect:
                first_connect = False
                # Start a watch
                await ws.send(json.dumps(make_event("node.invoke.request", {
                    "id": "w1", "nodeId": "test-node",
                    "command": "vision.watch",
                    "params": {"question": "test?", "interval": 60},
                })))
                # Wait for ack, then disconnect
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    pass
                await ws.close()
            else:
                # Second connection — signal test to check watches
                reconnected.set()
                try:
                    async for _ in ws:
                        pass
                except websockets.ConnectionClosed:
                    pass

        engine = MagicMock()
        with patch.object(CommandHandler, "_capture_frame", return_value=_fake_frame()):
            async with serve(mock_gateway, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                handler = CommandHandler(engine=engine, camera_sources=["0"])
                node = ClawNode(
                    gateway_url=f"ws://127.0.0.1:{port}",
                    node_id="test-node",
                    handler=handler,
                )
                node.token = "tok"

                task = asyncio.create_task(node.run())
                await asyncio.wait_for(reconnected.wait(), timeout=10)

                # After reconnect, watches from first connection should be cleared
                assert len(handler._watches) == 0

                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()


# =============================================================================
# _detect_triggered unit tests
# =============================================================================

class TestDetectTriggeredClaw:
    def test_yes(self):
        assert _detect_triggered("Yes, there is a person.") is True

    def test_no(self):
        assert _detect_triggered("No, the area is clear.") is False

    def test_positive_pattern(self):
        assert _detect_triggered("There is a package on the porch.") is True

    def test_negative_pattern(self):
        assert _detect_triggered("There is no one at the door.") is False

    def test_ambiguous(self):
        assert _detect_triggered("The image shows a red square.") is None


# =============================================================================
# vision.watch + vision.watch.stop unit tests
# =============================================================================

class TestVisionWatch:
    @pytest.mark.asyncio
    async def test_watch_requires_question(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()
        req = InvokeRequest(id="1", node_id="n", command="vision.watch", params={})
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_watch_requires_engine(self):
        handler = CommandHandler(engine=None, camera_sources=["0"])
        handler._ws = AsyncMock()
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch",
            params={"question": "Are there people?"},
        )
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "ENGINE_NOT_LOADED"

    @pytest.mark.asyncio
    async def test_watch_requires_ws(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        # _ws is None by default
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch",
            params={"question": "Are there people?"},
        )
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "NO_WS"

    @pytest.mark.asyncio
    async def test_watch_starts_and_returns_id(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch",
            params={"question": "Are there people?", "interval": 5},
        )
        result = await handler.handle(req)
        assert result.ok
        assert "watchId" in result.payload
        assert result.payload["status"] == "started"
        assert result.payload["interval"] == 5.0

        # Cleanup: stop the watch
        watch_id = result.payload["watchId"]
        assert watch_id in handler._watches
        stop_req = InvokeRequest(
            id="2", node_id="n", command="vision.watch.stop",
            params={"watchId": watch_id},
        )
        stop_result = await handler.handle(stop_req)
        assert stop_result.ok
        assert stop_result.payload["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_watch_interval_floor(self):
        """Interval is floored at 2s to prevent abuse."""
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch",
            params={"question": "test", "interval": 0.5},
        )
        result = await handler.handle(req)
        assert result.ok
        assert result.payload["interval"] == 2.0

        # Cleanup
        for w in list(handler._watches.values()):
            w.stop_event.set()
            if w.task:
                w.task.cancel()

    @pytest.mark.asyncio
    async def test_watch_stop_all(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()

        # Start two watches
        for i in range(2):
            req = InvokeRequest(
                id=str(i), node_id="n", command="vision.watch",
                params={"question": f"q{i}"},
            )
            await handler.handle(req)
        assert len(handler._watches) == 2

        # Stop all (no watchId param)
        stop_req = InvokeRequest(
            id="stop", node_id="n", command="vision.watch.stop", params={},
        )
        result = await handler.handle(stop_req)
        assert result.ok
        assert len(result.payload["stopped"]) == 2
        assert len(handler._watches) == 0

    @pytest.mark.asyncio
    async def test_watch_stop_not_found(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch.stop",
            params={"watchId": "nonexistent"},
        )
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_watch_max_limit(self):
        """Cannot exceed max concurrent watches."""
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()
        handler._max_watches = 2

        # Start 2 watches (at limit)
        for i in range(2):
            req = InvokeRequest(
                id=str(i), node_id="n", command="vision.watch",
                params={"question": f"q{i}"},
            )
            result = await handler.handle(req)
            assert result.ok

        # Third should fail
        req = InvokeRequest(
            id="3", node_id="n", command="vision.watch",
            params={"question": "q3"},
        )
        result = await handler.handle(req)
        assert not result.ok
        assert result.error_code == "LIMIT_REACHED"

        # Cleanup
        for w in list(handler._watches.values()):
            w.stop_event.set()
            if w.task:
                w.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_all_watches_cleanup(self):
        """stop_all_watches clears all watches (disconnect path)."""
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()

        for i in range(3):
            req = InvokeRequest(
                id=str(i), node_id="n", command="vision.watch",
                params={"question": f"q{i}"},
            )
            await handler.handle(req)
        assert len(handler._watches) == 3

        await handler.stop_all_watches()
        assert len(handler._watches) == 0

    @pytest.mark.asyncio
    async def test_vision_status_includes_watches(self):
        handler = CommandHandler(engine=MagicMock(), camera_sources=["0"])
        handler._ws = AsyncMock()

        # Start a watch
        req = InvokeRequest(
            id="1", node_id="n", command="vision.watch",
            params={"question": "test"},
        )
        await handler.handle(req)

        # Check status
        status_req = InvokeRequest(id="2", node_id="n", command="vision.status")
        result = await handler.handle(status_req)
        assert result.ok
        assert len(result.payload["watches"]) == 1
        assert result.payload["watches"][0]["condition"] == "test"

        # Cleanup status test
        for w in list(handler._watches.values()):
            w.stop_event.set()
            if w.task:
                w.task.cancel()


# =============================================================================
# RTSP failure escalation tests
# =============================================================================

class TestCaptureFailureEscalation:
    @pytest.mark.asyncio
    async def test_consecutive_failures_trigger_warning(self):
        """After CAPTURE_WARN_THRESHOLD failures, camera.offline event is sent."""
        engine = _mock_engine("Yes.")
        handler = CommandHandler(engine=engine, camera_sources=["0"])
        handler._ws = AsyncMock()
        handler._ws.send = AsyncMock(return_value=None)

        # Track what gets sent via WebSocket
        sent_frames = []
        original_send = handler._ws.send

        async def capture_send(data):
            sent_frames.append(json.loads(data))

        handler._ws.send = capture_send

        # Simulate capture returning None repeatedly
        call_count = 0

        def failing_capture(source):
            nonlocal call_count
            call_count += 1
            if call_count <= CAPTURE_WARN_THRESHOLD + 1:
                return None
            # After warning threshold + 1, return a real frame to stop the loop naturally
            return _fake_frame()

        with patch.object(handler, "_capture_frame", side_effect=failing_capture):
            watch = WatchTask(watch_id="test-w", condition="test?", interval=0.01)
            handler._watches["test-w"] = watch

            # Run the watch loop briefly
            task = asyncio.create_task(
                handler._watch_loop(watch, "0", "inv-1")
            )
            # Give it time to hit the threshold
            await asyncio.sleep(0.5)
            watch.stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        # Check that a camera.offline event was sent
        offline_events = [
            f for f in sent_frames
            if f.get("method") == "node.event"
            and f.get("params", {}).get("event") == "camera.offline"
        ]
        assert len(offline_events) >= 1

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_watch(self):
        """After CAPTURE_STOP_THRESHOLD failures, the watch stops itself."""
        engine = _mock_engine("Yes.")
        handler = CommandHandler(engine=engine, camera_sources=["0"])
        handler._ws = AsyncMock()

        sent_frames = []

        async def capture_send(data):
            sent_frames.append(json.loads(data))

        handler._ws.send = capture_send

        # Always fail
        with patch.object(handler, "_capture_frame", return_value=None):
            watch = WatchTask(watch_id="test-w2", condition="test?", interval=0.01)
            handler._watches["test-w2"] = watch

            task = asyncio.create_task(
                handler._watch_loop(watch, "0", "inv-2")
            )
            # Wait for it to hit the stop threshold and exit
            await asyncio.wait_for(task, timeout=10)

        # Watch should have self-terminated
        assert "test-w2" not in handler._watches

    @pytest.mark.asyncio
    async def test_failure_counter_resets_on_success(self):
        """Successful capture resets the consecutive failure counter."""
        engine = _mock_engine("No.")
        handler = CommandHandler(engine=engine, camera_sources=["0"])
        handler._ws = AsyncMock()
        handler._ws.send = AsyncMock(return_value=None)

        call_count = 0

        def intermittent_capture(source):
            nonlocal call_count
            call_count += 1
            # Fail 3 times, then succeed
            if call_count % 4 != 0:
                return None
            return _fake_frame()

        with patch.object(handler, "_capture_frame", side_effect=intermittent_capture):
            watch = WatchTask(watch_id="test-w3", condition="test?", interval=0.01)
            handler._watches["test-w3"] = watch

            task = asyncio.create_task(
                handler._watch_loop(watch, "0", "inv-3")
            )
            # Let it run a few cycles
            await asyncio.sleep(0.5)
            watch.stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        # Failures never accumulated past 3 (reset each success),
        # so metrics should show some failures but watch shouldn't have auto-stopped
        assert handler._metrics.capture_failures > 0
        assert watch.consecutive_failures < CAPTURE_WARN_THRESHOLD


# =============================================================================
# Full watch loop integration test (mock Gateway + mock engine + mock camera)
# =============================================================================

def _fake_frame():
    """Return a 64x64 BGR frame."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _mock_engine(answer="Yes, there is a person."):
    """Return a mock engine whose analyze_frame returns a fixed answer."""
    engine = MagicMock()
    result_obj = MagicMock()
    result_obj.text = answer
    engine.analyze_frame.return_value = result_obj
    engine.health.return_value = {"backend": {"model": "mock", "device": "cpu", "backend": "mock"}}
    return engine


class TestWatchLoopIntegration:
    """End-to-end: mock Gateway sends vision.watch invoke, receives streaming
    results, then sends vision.watch.stop and verifies summary."""

    @pytest.mark.asyncio
    async def test_watch_loop_streams_results_and_stops(self):
        """Start watch → receive ≥2 streaming results → stop → verify summary."""
        streaming_results = []
        start_result = None
        stop_result = None
        done = asyncio.Event()

        async def mock_gateway(ws):
            nonlocal start_result, stop_result

            # --- Handshake ---
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n1"}),
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # --- Send vision.watch invoke ---
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "watch-inv-1",
                "nodeId": "test-node",
                "command": "vision.watch",
                "params": {"question": "Is there a person?", "interval": 2},
            })))

            # --- Collect frames ---
            # First frame: the start acknowledgement (node.invoke.result with watchId)
            # Subsequent frames: streaming watch results
            try:
                while not done.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("method") != "node.invoke.result":
                        continue
                    params = msg["params"]
                    payload = json.loads(params.get("payloadJSON", "{}"))

                    if "watchId" in payload and "status" in payload and payload["status"] == "started":
                        start_result = payload
                    elif params.get("streaming"):
                        streaming_results.append(payload)
                        # After 2 streaming results, send stop
                        if len(streaming_results) >= 2:
                            watch_id = payload["watchId"]
                            await ws.send(json.dumps(make_event("node.invoke.request", {
                                "id": "stop-inv-1",
                                "nodeId": "test-node",
                                "command": "vision.watch.stop",
                                "params": {"watchId": watch_id},
                            })))
                    elif "status" in payload and payload["status"] == "stopped":
                        stop_result = payload
                        done.set()
            except asyncio.TimeoutError:
                done.set()

        engine = _mock_engine("Yes, there is a person.")

        # Patch _capture_frame to return a fake frame without opening a camera
        with patch.object(CommandHandler, "_capture_frame", return_value=_fake_frame()):
            async with serve(mock_gateway, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                handler = CommandHandler(engine=engine, camera_sources=["0"])
                node = ClawNode(
                    gateway_url=f"ws://127.0.0.1:{port}",
                    node_id="test-node",
                    handler=handler,
                )
                node.token = "test-token"

                task = asyncio.create_task(node.run())
                await asyncio.wait_for(done.wait(), timeout=15)
                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

        # --- Assertions ---
        # Start ack received
        assert start_result is not None
        assert start_result["status"] == "started"
        assert start_result["interval"] == 2.0
        watch_id = start_result["watchId"]

        # Got streaming results
        assert len(streaming_results) >= 2
        for sr in streaming_results:
            assert sr["watchId"] == watch_id
            assert sr["answer"] == "Yes, there is a person."
            assert sr["triggered"] is True
            assert "frame" in sr  # frame included on triggered=True
            assert sr["latency_ms"] >= 0

        # Counters increment
        assert streaming_results[0]["checks"] == 1
        assert streaming_results[1]["checks"] == 2

        # Stop result received
        assert stop_result is not None
        assert stop_result["status"] == "stopped"
        assert stop_result["watchId"] == watch_id
        assert stop_result["checks"] >= 2
        assert stop_result["alerts"] >= 2

    @pytest.mark.asyncio
    async def test_watch_no_alert_no_frame(self):
        """When triggered=False, streaming results should NOT include a frame."""
        got_result = asyncio.Event()
        streaming_result = None

        async def mock_gateway(ws):
            nonlocal streaming_result
            # Handshake
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n2"}),
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # Send vision.watch
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "w2",
                "nodeId": "test-node",
                "command": "vision.watch",
                "params": {"question": "Is it raining?", "interval": 2},
            })))

            # Collect: skip start ack, grab first streaming result
            try:
                while not got_result.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("method") != "node.invoke.result":
                        continue
                    params = msg["params"]
                    if params.get("streaming"):
                        streaming_result = json.loads(params["payloadJSON"])
                        got_result.set()
            except asyncio.TimeoutError:
                got_result.set()

        engine = _mock_engine("No, it is not raining.")

        with patch.object(CommandHandler, "_capture_frame", return_value=_fake_frame()):
            async with serve(mock_gateway, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                handler = CommandHandler(engine=engine, camera_sources=["0"])
                node = ClawNode(
                    gateway_url=f"ws://127.0.0.1:{port}",
                    node_id="test-node",
                    handler=handler,
                )
                node.token = "test-token"

                task = asyncio.create_task(node.run())
                await asyncio.wait_for(got_result.wait(), timeout=10)
                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

        assert streaming_result is not None
        assert streaming_result["triggered"] is False
        assert streaming_result["alerts"] == 0
        assert "frame" not in streaming_result  # no frame when not triggered

    @pytest.mark.asyncio
    async def test_watch_disconnect_cleanup(self):
        """When Gateway disconnects, watches should be cleaned up."""
        got_streaming = asyncio.Event()

        async def mock_gateway(ws):
            # Handshake
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "n3"}),
            }))
            frame = json.loads(await ws.recv())
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # Start a watch
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "w3",
                "nodeId": "test-node",
                "command": "vision.watch",
                "params": {"question": "test?", "interval": 2},
            })))

            # Wait for at least one streaming result then close abruptly
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    params = msg.get("params", {})
                    if params.get("streaming"):
                        got_streaming.set()
                        break
            except asyncio.TimeoutError:
                pass
            # Close WebSocket — node should clean up watches
            await ws.close()

        engine = _mock_engine("Yes.")

        with patch.object(CommandHandler, "_capture_frame", return_value=_fake_frame()):
            async with serve(mock_gateway, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                handler = CommandHandler(engine=engine, camera_sources=["0"])
                node = ClawNode(
                    gateway_url=f"ws://127.0.0.1:{port}",
                    node_id="test-node",
                    handler=handler,
                )
                node.token = "test-token"

                task = asyncio.create_task(node.run())
                await asyncio.wait_for(got_streaming.wait(), timeout=10)
                # Give node time to detect disconnect and clean up
                await asyncio.sleep(0.5)
                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

        # Watches should be cleaned up after disconnect
        assert len(handler._watches) == 0


# =============================================================================
# Health server tests
# =============================================================================

class TestHealthServer:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health returns JSON status."""
        from trio_core.claw.health import HealthServer

        handler = CommandHandler(engine=None, camera_sources=["0"])
        node = ClawNode(gateway_url="ws://localhost:1234", handler=handler)
        node.token = "tok"
        node._start_time = 100.0

        server = HealthServer(node, port=0)
        # Use a random port
        raw_server = await asyncio.start_server(
            server._handle_connection, "127.0.0.1", 0,
        )
        port = raw_server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = await asyncio.wait_for(reader.read(4096), timeout=3)
            response_str = response.decode()

            assert "200 OK" in response_str
            assert "application/json" in response_str
            # Parse the JSON body
            body_start = response_str.index("\r\n\r\n") + 4
            body = json.loads(response_str[body_start:])
            assert "status" in body
            assert "uptime_s" in body
            assert "watches_active" in body

            writer.close()
            await writer.wait_closed()
        finally:
            raw_server.close()
            await raw_server.wait_closed()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """GET /metrics returns Prometheus text format."""
        from trio_core.claw.health import HealthServer

        handler = CommandHandler(engine=None, camera_sources=["0"])
        handler._metrics.watch_checks = 42
        handler._metrics.capture_failures = 3
        node = ClawNode(gateway_url="ws://localhost:1234", handler=handler)
        node.token = "tok"
        node._start_time = 100.0

        server = HealthServer(node, port=0)
        raw_server = await asyncio.start_server(
            server._handle_connection, "127.0.0.1", 0,
        )
        port = raw_server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = await asyncio.wait_for(reader.read(4096), timeout=3)
            response_str = response.decode()

            assert "200 OK" in response_str
            assert "trio_claw_connected 0" in response_str  # not connected
            assert "trio_claw_watch_checks_total 42" in response_str
            assert "trio_claw_capture_failures_total 3" in response_str

            writer.close()
            await writer.wait_closed()
        finally:
            raw_server.close()
            await raw_server.wait_closed()

    @pytest.mark.asyncio
    async def test_404_on_unknown_path(self):
        """Unknown paths return 404."""
        from trio_core.claw.health import HealthServer

        handler = CommandHandler(engine=None, camera_sources=["0"])
        node = ClawNode(gateway_url="ws://localhost:1234", handler=handler)
        node.token = "tok"

        server = HealthServer(node, port=0)
        raw_server = await asyncio.start_server(
            server._handle_connection, "127.0.0.1", 0,
        )
        port = raw_server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = await asyncio.wait_for(reader.read(4096), timeout=3)
            assert b"404 Not Found" in response

            writer.close()
            await writer.wait_closed()
        finally:
            raw_server.close()
            await raw_server.wait_closed()


# =============================================================================
# Smoke test — full lifecycle end-to-end
# =============================================================================

class TestClawSmoke:
    """End-to-end smoke test: mock Gateway → handshake → camera.list →
    vision.analyze → vision.watch (1 cycle) → health endpoint → shutdown.

    Exercises every production code path without real hardware.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        results = {}
        phase = asyncio.Event()
        phases_done = {"list": False, "analyze": False, "watch_result": False}

        async def mock_gateway(ws):
            # --- Handshake ---
            await ws.send(json.dumps({
                "type": "event", "event": "connect.challenge",
                "payloadJSON": json.dumps({"nonce": "smoke-nonce"}),
            }))
            frame = json.loads(await ws.recv())
            assert frame["method"] == "connect"
            assert frame["params"]["role"] == "node"
            await ws.send(json.dumps({
                "type": "res", "id": frame["id"], "ok": True,
                "payload": {"type": "hello-ok"},
            }))

            # --- Phase 1: camera.list ---
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "smoke-list", "nodeId": "smoke-node",
                "command": "camera.list", "params": {},
            })))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            payload = json.loads(msg["params"]["payloadJSON"])
            results["list"] = payload
            phases_done["list"] = True

            # --- Phase 2: vision.analyze ---
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "smoke-analyze", "nodeId": "smoke-node",
                "command": "vision.analyze",
                "params": {"question": "What do you see?"},
            })))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            payload = json.loads(msg["params"]["payloadJSON"])
            results["analyze"] = payload
            phases_done["analyze"] = True

            # --- Phase 3: vision.watch (collect 1 streaming result, then stop) ---
            await ws.send(json.dumps(make_event("node.invoke.request", {
                "id": "smoke-watch", "nodeId": "smoke-node",
                "command": "vision.watch",
                "params": {"question": "Is anyone there?", "interval": 2},
            })))
            # Collect start ack + first streaming result
            watch_id = None
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("method") != "node.invoke.result":
                        continue
                    params = msg["params"]
                    p = json.loads(params.get("payloadJSON", "{}"))
                    if p.get("status") == "started":
                        watch_id = p["watchId"]
                        results["watch_start"] = p
                    elif params.get("streaming"):
                        results["watch_result"] = p
                        phases_done["watch_result"] = True
                        # Stop the watch
                        await ws.send(json.dumps(make_event("node.invoke.request", {
                            "id": "smoke-stop", "nodeId": "smoke-node",
                            "command": "vision.watch.stop",
                            "params": {"watchId": watch_id},
                        })))
                    elif p.get("status") == "stopped":
                        results["watch_stop"] = p
                        break
            except asyncio.TimeoutError:
                pass

            phase.set()
            # Keep connection open for health check
            try:
                async for _ in ws:
                    pass
            except websockets.ConnectionClosed:
                pass

        engine = _mock_engine("Yes, I see a person walking.")

        with patch.object(CommandHandler, "_capture_frame", return_value=_fake_frame()):
            async with serve(mock_gateway, "127.0.0.1", 0) as gw_server:
                gw_port = gw_server.sockets[0].getsockname()[1]

                handler = CommandHandler(engine=engine, camera_sources=["0", "rtsp://test"])
                node = ClawNode(
                    gateway_url=f"ws://127.0.0.1:{gw_port}",
                    node_id="smoke-node",
                    handler=handler,
                )
                node.token = "smoke-token"

                # Start health server
                from trio_core.claw.health import HealthServer
                health_srv = await asyncio.start_server(
                    HealthServer(node)._handle_connection, "127.0.0.1", 0,
                )
                health_port = health_srv.sockets[0].getsockname()[1]

                # Run node
                task = asyncio.create_task(node.run())
                await asyncio.wait_for(phase.wait(), timeout=20)

                # --- Phase 4: Health endpoint ---
                reader, writer = await asyncio.open_connection("127.0.0.1", health_port)
                writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                resp = await asyncio.wait_for(reader.read(4096), timeout=3)
                resp_str = resp.decode()
                body_start = resp_str.index("\r\n\r\n") + 4
                health_body = json.loads(resp_str[body_start:])
                results["health"] = health_body
                writer.close()
                await writer.wait_closed()

                # --- Phase 5: Metrics endpoint ---
                reader, writer = await asyncio.open_connection("127.0.0.1", health_port)
                writer.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                resp = await asyncio.wait_for(reader.read(4096), timeout=3)
                results["metrics"] = resp.decode()
                writer.close()
                await writer.wait_closed()

                # --- Phase 6: Graceful shutdown ---
                await node.stop()
                try:
                    await asyncio.wait_for(task, timeout=3)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

                health_srv.close()
                await health_srv.wait_closed()

        # =================================================================
        # Assertions — verify every phase
        # =================================================================

        # Phase 1: camera.list
        assert "devices" in results["list"]
        assert len(results["list"]["devices"]) == 2
        assert results["list"]["devices"][0]["id"] == "cam-0"

        # Phase 2: vision.analyze
        assert results["analyze"]["answer"] == "Yes, I see a person walking."
        assert results["analyze"]["latency_ms"] >= 0
        assert "frame" in results["analyze"]
        assert results["analyze"]["frame"]["format"] == "jpeg"

        # Phase 3: vision.watch
        assert results["watch_start"]["status"] == "started"
        assert results["watch_result"]["triggered"] is True
        assert results["watch_result"]["checks"] == 1
        assert "frame" in results["watch_result"]  # triggered=True → includes frame
        assert results["watch_stop"]["status"] == "stopped"

        # Phase 4: Health
        assert results["health"]["status"] == "connected"
        assert results["health"]["uptime_s"] >= 0
        assert results["health"]["model"] == "mock"

        # Phase 5: Metrics
        assert "trio_claw_connected 1" in results["metrics"]
        assert "trio_claw_watch_checks_total" in results["metrics"]
        assert "trio_claw_vlm_latency_seconds" in results["metrics"]

        # Phase 6: Shutdown was clean (task completed)
        assert task.done()

        # Metrics accumulated correctly
        assert handler._metrics.watch_checks >= 1
        assert handler._metrics.watch_alerts >= 1
        assert len(handler._metrics.vlm_latency_samples) >= 2  # analyze + watch
