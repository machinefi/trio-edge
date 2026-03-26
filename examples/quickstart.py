"""TrioCore quickstart — analyze an image in 5 lines."""

from trio_core import TrioCore

engine = TrioCore()
engine.load()

result = engine.analyze_video("photo.jpg", "What do you see?")
print(result.text)
print(f"{result.metrics.latency_ms:.0f}ms | {result.metrics.tokens_per_sec:.0f} tok/s")
