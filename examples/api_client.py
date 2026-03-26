"""Call the TrioCore inference API from Python.

Start the server first:
    trio serve

Then run this script:
    python examples/api_client.py photo.jpg
"""

from __future__ import annotations

import base64
import json
import sys

import urllib.request

API_BASE = "http://localhost:8100"


def detect(image_path: str) -> dict:
    """Run YOLO detection on an image via the API."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({"image_b64": image_b64}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/inference/detect",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def describe(image_path: str, prompt: str = "Describe what you see.") -> dict:
    """Run VLM description on an image via the API."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({"image_b64": image_b64, "prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/inference/describe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/api_client.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=== Detection ===")
    result = detect(image_path)
    print(f"People: {result['people_count']}, Vehicles: {result['vehicle_count']}")
    print(f"By class: {result['by_class']}")
    print(f"Elapsed: {result['elapsed_ms']}ms")

    print("\n=== Description ===")
    result = describe(image_path)
    print(f"Description: {result['description']}")
    print(f"Elapsed: {result['elapsed_ms']}ms")
