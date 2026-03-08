#!/bin/bash
# trio-core API examples — copy-paste ready curl commands.
#
# Start the server first:
#   trio serve
#   # or: trio serve --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
#
# The server runs at http://localhost:8000 by default.

BASE="http://localhost:8000"

echo "=== Health Check ==="
curl -s "$BASE/health" | python3 -m json.tool
echo

echo "=== Analyze a single frame (base64 JPEG) ==="
# Encode an image to base64 and send it
IMAGE_B64=$(python3 -c "
import base64, io
from PIL import Image
img = Image.new('RGB', (640, 480), (100, 150, 200))
buf = io.BytesIO()
img.save(buf, format='JPEG')
print(base64.b64encode(buf.getvalue()).decode())
")

curl -s "$BASE/analyze-frame" \
  -H "Content-Type: application/json" \
  -d "{\"frame_b64\": \"$IMAGE_B64\", \"question\": \"What color is this image?\"}" \
  | python3 -m json.tool
echo

echo "=== Chat Completions (OpenAI-compatible) ==="
# Works with any OpenAI client — just change the base URL.
# Supports: image_url (file path, URL, or base64 data URI)
curl -s "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
          {"type": "text", "text": "What do you see in this image?"}
        ]
      }
    ],
    "max_tokens": 128
  }' | python3 -m json.tool
echo

echo "=== Streaming (SSE) ==="
# Same endpoint with stream=true — get tokens as they're generated.
curl -s -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
          {"type": "text", "text": "Describe this image briefly."}
        ]
      }
    ],
    "max_tokens": 64
  }'
echo

echo "=== Analyze a local video file ==="
curl -s "$BASE/v1/video/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video": "test_videos/intruder_house.mp4",
    "prompt": "Is there a person? What are they doing?",
    "max_tokens": 128
  }' | python3 -m json.tool
echo

echo "=== Upload frames directly (multipart) ==="
# Create a test image file
python3 -c "from PIL import Image; Image.new('RGB', (320, 240), (255, 0, 0)).save('/tmp/trio_test.jpg')"

curl -s "$BASE/v1/frames/analyze" \
  -F "prompt=What color is this?" \
  -F "max_tokens=32" \
  -F "frames=@/tmp/trio_test.jpg" \
  | python3 -m json.tool

rm -f /tmp/trio_test.jpg
echo

echo "Done. See https://github.com/machinefi/trio-core for full docs."
