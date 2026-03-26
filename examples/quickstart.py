"""TrioCore quickstart — analyze an image in 5 lines.

Usage:
    python examples/quickstart.py              # uses sample.jpg in current dir
    python examples/quickstart.py myimage.jpg  # uses your own image

First run will download the default model (~2 GB). This is a one-time download.
"""

import sys

from trio_core import TrioCore

# Use the image path from command line, or default to "sample.jpg"
image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

engine = TrioCore()
engine.load()  # Downloads model on first run (~2 GB, takes a few minutes)

result = engine.analyze_video(image_path, "What do you see?")
print(result.text)
print(f"{result.metrics.latency_ms:.0f}ms | {result.metrics.tokens_per_sec:.0f} tok/s")
