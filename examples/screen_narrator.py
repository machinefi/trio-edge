#!/usr/bin/env python3
"""Screen Narrator — VLM watches your screen and explains what you're doing.

Captures a specific app window (default: Chrome) and asks the VLM to describe
your activity. Shows an overlay window with the narration.

Usage:
    python examples/screen_narrator.py
    python examples/screen_narrator.py --app Safari
    python examples/screen_narrator.py --app none  # full screen
    python examples/screen_narrator.py --interval 5

Requires:
    pip install opencv-python pyobjc-framework-Quartz
    pip install 'trio-core[mlx]' or pip install 'trio-core[transformers]'

Controls:
    q / ESC — quit
    SPACE   — force re-analyze now
"""

import argparse
import ctypes
import threading
import time

import cv2
import numpy as np


def _cg_image_to_numpy(cg_image) -> np.ndarray | None:
    """Convert a CGImage to a BGR numpy array."""
    import Quartz
    from CoreFoundation import CFDataGetBytes, CFDataGetLength

    w = Quartz.CGImageGetWidth(cg_image)
    h = Quartz.CGImageGetHeight(cg_image)
    if w == 0 or h == 0:
        return None

    # Get raw pixel data
    data_provider = Quartz.CGImageGetDataProvider(cg_image)
    cf_data = Quartz.CGDataProviderCopyData(data_provider)
    length = CFDataGetLength(cf_data)

    buf = ctypes.create_string_buffer(length)
    CFDataGetBytes(cf_data, (0, length), buf)

    # BGRA format from CoreGraphics
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[:, :, :3].copy()  # drop alpha, BGR


def find_and_capture_app(
    app_name: str, scale: float = 0.5
) -> tuple[np.ndarray | None, dict | None]:
    """Find app window and capture its screen region. Returns (frame_bgr, bounds)."""
    import Quartz

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )

    best_area = 0
    best_bounds = None
    for w in windows:
        owner = w.get("kCGWindowOwnerName", "")
        if app_name.lower() not in owner.lower():
            continue
        bounds = w.get("kCGWindowBounds", {})
        area = bounds.get("Width", 0) * bounds.get("Height", 0)
        if area > best_area and area > 10000:
            best_area = area
            w.get("kCGWindowNumber")
            best_bounds = bounds

    if best_bounds is None:
        return None, None

    rect = Quartz.CGRectMake(
        float(best_bounds["X"]),
        float(best_bounds["Y"]),
        float(best_bounds["Width"]),
        float(best_bounds["Height"]),
    )

    # Capture the screen region where this window is
    cg_image = Quartz.CGWindowListCreateImage(
        rect,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault,
    )

    if cg_image is None:
        return None, best_bounds

    frame = _cg_image_to_numpy(cg_image)
    if frame is None:
        return None, best_bounds

    if scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    info = {
        "left": int(best_bounds["X"]),
        "top": int(best_bounds["Y"]),
        "width": int(best_bounds["Width"]),
        "height": int(best_bounds["Height"]),
    }
    return frame, info


def capture_full_screen(scale: float = 0.5) -> np.ndarray | None:
    """Capture the full screen."""
    import Quartz

    cg_image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectInfinite,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault,
    )
    if cg_image is None:
        return None

    frame = _cg_image_to_numpy(cg_image)
    if frame is None:
        return None

    if scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return frame


def draw_overlay(frame: np.ndarray, text: str, metrics: str = "") -> np.ndarray:
    """Draw narration text overlay at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    margin = 10
    max_width = w - 2 * margin

    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    if metrics:
        lines.append(metrics)

    if not lines:
        return frame

    line_height = 20
    box_height = len(lines) * line_height + 2 * margin
    box_top = h - box_height

    cv2.rectangle(overlay, (0, box_top), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    for i, line in enumerate(lines):
        y = box_top + margin + (i + 1) * line_height - 4
        color = (160, 160, 160) if i == len(lines) - 1 and metrics else (255, 255, 255)
        cv2.putText(frame, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Screen Narrator — VLM watches your screen")
    parser.add_argument(
        "--app",
        "-a",
        default="Chrome",
        help="App window to capture (default: Chrome). Use 'none' for full screen.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Look at this screenshot of a web browser. Describe what the user is doing — what website or page are they on, and what are they reading, writing, or interacting with? Be specific in 1-2 sentences.",
        help="Question for the VLM",
    )
    parser.add_argument("--model", "-m", default=None, help="Model name")
    parser.add_argument("--backend", "-b", default=None, help="Force backend: mlx or transformers")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max generation tokens")
    parser.add_argument(
        "--interval", type=float, default=3.0, help="Seconds between analyses (default: 3.0)"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Scale factor for capture (default: 0.5)"
    )
    args = parser.parse_args()

    use_app = args.app.lower() != "none"

    # Initialize engine
    from trio_core import EngineConfig, TrioCore

    config_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        config_kwargs["model"] = args.model
    config = EngineConfig(**config_kwargs)

    engine = TrioCore(config, backend=args.backend)
    print(f"Loading model: {config.model} ...")
    engine.load()
    health = engine.health()
    print(f"Backend: {health.get('backend', {}).get('backend', 'unknown')}")
    print(f"Device: {health.get('backend', {}).get('device', 'unknown')}")

    # Test capture
    if use_app:
        frame, info = find_and_capture_app(args.app, args.scale)
        if frame is not None:
            print(
                f"Found {args.app} window: {info['width']}x{info['height']} at ({info['left']}, {info['top']})"
            )
            print(f"Captured frame: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Warning: {args.app} window not found, falling back to full screen")
            use_app = False

    print(f"\nCapturing every {args.interval}s at {args.scale:.0%} scale")

    # Shared state
    lock = threading.Lock()
    narration = "Analyzing your screen..."
    metrics_text = ""
    force_analyze = threading.Event()
    running = True

    def inference_loop():
        nonlocal narration, metrics_text, running

        while running:
            force_analyze.wait(timeout=args.interval)
            force_analyze.clear()

            if not running:
                break

            # Capture
            if use_app:
                frame_bgr, _ = find_and_capture_app(args.app, args.scale)
            else:
                frame_bgr = capture_full_screen(args.scale)

            if frame_bgr is None:
                with lock:
                    narration = f"Cannot capture {args.app} — window not found"
                continue

            # BGR uint8 -> RGB float32
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_float = rgb.astype(np.float32) / 255.0

            try:
                t0 = time.monotonic()
                result = engine.analyze_frame(rgb_float, args.prompt, max_tokens=args.max_tokens)
                elapsed = time.monotonic() - t0

                with lock:
                    narration = result.text.strip().replace("\n", " ")
                    metrics_text = (
                        f"[{elapsed:.1f}s | "
                        f"prefill {result.metrics.preprocess_ms:.0f}ms | "
                        f"inference {result.metrics.inference_ms:.0f}ms | "
                        f"{result.metrics.tokens_per_sec:.1f} tok/s]"
                    )
                print(f"\n> {narration}")
                print(f"  {metrics_text}")
            except Exception as e:
                with lock:
                    narration = f"Error: {e}"
                print(f"Inference error: {e}")

    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    window_name = f"TrioCore Screen Narrator — {args.app if use_app else 'Full Screen'}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("\nPress 'q' or ESC to quit, SPACE to force re-analyze.")

    try:
        while True:
            if use_app:
                frame, _ = find_and_capture_app(args.app, args.scale)
            else:
                frame = capture_full_screen(args.scale)

            if frame is None:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)

            with lock:
                display = draw_overlay(frame, narration, metrics_text)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                force_analyze.set()
    finally:
        running = False
        force_analyze.set()
        thread.join(timeout=3)
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
