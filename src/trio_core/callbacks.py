"""Lightweight callback/hook system inspired by ultralytics.

Usage:
    engine = TrioCore()
    engine.add_callback("on_vlm_end", lambda e: print(f"VLM done: {e.last_result.text}"))

Event lifecycle:
    on_engine_load       → model loaded
    on_frame_captured    → raw frames extracted from source
    on_dedup_done        → temporal dedup complete
    on_motion_check      → motion gate evaluated
    on_vlm_start         → about to call VLM
    on_vlm_end           → VLM response received
    on_trigger           → watch condition met (for monitoring)
    on_stream_start      → StreamCapture opened
    on_stream_frame      → new frame from live stream
    on_stream_stop       → StreamCapture closed
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

logger = logging.getLogger(__name__)

# All recognized event names
EVENTS = (
    "on_engine_load",
    "on_frame_captured",
    "on_dedup_done",
    "on_motion_check",
    "on_vlm_start",
    "on_vlm_end",
    "on_trigger",
    "on_stream_start",
    "on_stream_frame",
    "on_stream_stop",
)


def get_default_callbacks() -> dict[str, list[Callable]]:
    """Return a fresh callback dict with empty lists for all events."""
    return defaultdict(list, {e: [] for e in EVENTS})


class CallbackMixin:
    """Mixin that adds callback support to any class.

    Classes using this mixin get:
        self.callbacks: dict[str, list[Callable]]
        self.add_callback(event, fn)
        self.run_callbacks(event)
        self.clear_callbacks(event=None)
    """

    def _init_callbacks(self, callbacks: dict[str, list[Callable]] | None = None) -> None:
        self.callbacks = callbacks or get_default_callbacks()

    def add_callback(self, event: str, fn: Callable) -> None:
        """Register a callback for an event."""
        if event not in EVENTS:
            logger.warning("Unknown callback event: %s (known: %s)", event, ", ".join(EVENTS))
        self.callbacks[event].append(fn)

    def run_callbacks(self, event: str) -> None:
        """Fire all callbacks for an event, passing self as the argument."""
        for fn in self.callbacks.get(event, []):
            try:
                fn(self)
            except Exception:
                logger.exception("Callback error in %s", event)

    def clear_callbacks(self, event: str | None = None) -> None:
        """Clear callbacks for a specific event, or all events if None."""
        if event:
            self.callbacks[event] = []
        else:
            self.callbacks = get_default_callbacks()
