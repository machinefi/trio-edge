"""VLM backend abstraction — one interface, multiple runtimes.

Backends:
    MLXBackend          → mlx-vlm on Apple Silicon (Metal GPU)
    TransformersBackend → HuggingFace Transformers on CUDA / CPU
    RemoteHTTPBackend   → OpenAI-compatible remote API
    ToMeMLXBackend      → MLX with Token Merging compression
    CompressedMLXBackend → MLX with visual token compression

Usage:
    from trio_core.backends import auto_backend

    backend = auto_backend("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    backend.load()
    result = backend.generate(frames, prompt, max_tokens=512)
"""

# isort: skip_file — re-export module, explicit ordering takes priority.

from trio_core.backends.base import (  # noqa: F401, I001
    BaseBackend,
    GenerationResult,
    StreamChunk,
    _TokenHandler as _TokenHandler,
)
from trio_core.backends.mlx import MLXBackend, compute_compressed_grid
from trio_core.backends.remote import RemoteHTTPBackend
from trio_core.backends.registry import (  # noqa: F401
    _BACKEND_MAP as _BACKEND_MAP,
    auto_backend,
    register_backend,
    resolve_backend,
)
from trio_core.backends.transformers import TransformersBackend

# Leaf backends are NOT imported here — they're loaded lazily by resolve_backend()
# to avoid importing optional deps (native_vision, token_compression) at startup.
# Users who need them directly can:
#   from trio_core.backends.tome import ToMeMLXBackend
#   from trio_core.backends.compressed import CompressedMLXBackend

__all__ = [
    "BaseBackend",
    "GenerationResult",
    "StreamChunk",
    "MLXBackend",
    "TransformersBackend",
    "RemoteHTTPBackend",
    "auto_backend",
    "resolve_backend",
    "register_backend",
    "compute_compressed_grid",
]
