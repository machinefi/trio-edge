"""Tests for OpenClaw node integration — protocol, handshake, invoke dispatch."""

import asyncio
import json

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
from trio_core.claw.node import ClawNode
from trio_core.claw.commands import CommandHandler


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
        assert params["client"]["id"] == "trio-core"
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

            try:
                await asyncio.wait_for(node.run(), timeout=3)
            except Exception:
                pass

        assert pong_received.is_set()
