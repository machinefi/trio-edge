#!/usr/bin/env python3
"""Quick test: invoke a command on a trio-core node via OpenClaw Gateway."""

import asyncio
import json
import sys

import websockets

from trio_core.claw.protocol import (
    make_req,
)

GATEWAY = "ws://127.0.0.1:18789"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
NODE_ID = sys.argv[2] if len(sys.argv) > 2 else ""
COMMAND = sys.argv[3] if len(sys.argv) > 3 else "camera.list"
PARAMS = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}


async def main():
    if not TOKEN:
        print("Usage: python test_invoke.py <gateway-token> <node-id> [command] [params-json]")
        print("Example: python test_invoke.py mytoken 11e600d228cc5f42 camera.list")
        return

    ws = await websockets.connect(GATEWAY)
    try:
        # 1. Challenge
        json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

        # 2. Connect as operator (no device identity, just gateway token)
        params = {
            "minProtocol": 3,
            "maxProtocol": 3,
            "client": {
                "id": "cli",
                "version": "0.1.0",
                "platform": "darwin",
                "deviceFamily": "trio-core",
                "modelIdentifier": "test",
                "mode": "cli",
            },
            "role": "operator",
            "scopes": [
                "operator.admin",
                "operator.write",
                "operator.read",
                "operator.approvals",
                "operator.pairing",
            ],
            "auth": {"token": TOKEN},
            "caps": [],
            "commands": [],
        }

        await ws.send(json.dumps(make_req("connect", params)))

        # 3. Hello
        res = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if not res.get("ok"):
            print(f"Connect failed: {res}")
            return
        print("Connected as operator")

        # 4. Send invoke
        invoke_id = "test-inv-1"
        invoke_req = make_req(
            "node.invoke",
            {
                "nodeId": NODE_ID,
                "command": COMMAND,
                "params": PARAMS,
                "idempotencyKey": f"test-{invoke_id}",
            },
        )
        invoke_req["id"] = invoke_id
        await ws.send(json.dumps(invoke_req))
        print(f"Sent invoke: {COMMAND} -> node {NODE_ID}")

        # 5. Wait for result
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)

            if msg.get("type") == "res" and msg.get("id") == invoke_id:
                print(f"\nResult (ok={msg.get('ok')}):")
                print(json.dumps(msg, indent=2, ensure_ascii=False)[:2000])
                break
            elif msg.get("type") == "event" and msg.get("event") == "node.invoke.result":
                payload = json.loads(msg.get("payloadJSON", "{}"))
                print("\nResult event:")
                print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])
                break
            # skip other events

    finally:
        await ws.close()


asyncio.run(main())
