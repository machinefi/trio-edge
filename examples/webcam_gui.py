#!/usr/bin/env python3
"""Webcam GUI — live VLM analysis with video preview and text overlay.

Shows a window with the webcam feed and VLM description overlaid at the bottom.
Inference runs in a background thread so the video stays smooth.

Usage:
    python examples/webcam_gui.py
    python examples/webcam_gui.py --source "rtsp://admin:pass@192.168.1.100:554/stream1"
    python examples/webcam_gui.py --source test_videos/intruder_house.mp4
    python examples/webcam_gui.py --prompt "What is the person doing?"
    python examples/webcam_gui.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit

Requires:
    pip install opencv-python   (NOT opencv-python-headless)
    pip install 'trio-core[mlx]' or pip install 'trio-core[transformers]'

Controls:
    q / ESC — quit
    SPACE   — force re-analyze current frame
"""

import argparse
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np


def draw_text_box(frame: np.ndarray, text: str, metrics: str = "") -> np.ndarray:
    """Draw a semi-transparent box at the bottom with text overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Wrap text to fit window width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    margin = 10
    max_width = w - 2 * margin

    # Simple word-wrap
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

    line_height = 22
    box_height = len(lines) * line_height + 2 * margin
    box_top = h - box_height

    # Semi-transparent black background
    cv2.rectangle(overlay, (0, box_top), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Draw text lines
    for i, line in enumerate(lines):
        y = box_top + margin + (i + 1) * line_height - 4
        color = (180, 180, 180) if i == len(lines) - 1 and metrics else (255, 255, 255)
        cv2.putText(frame, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Webcam GUI with live VLM analysis")
    parser.add_argument("--source", "-s", default="0",
                        help="RTSP URL, video file, or camera index (default: 0)")
    parser.add_argument("--prompt", "-p", default="Describe what you see in one sentence.",
                        help="Question to ask the VLM")
    parser.add_argument("--model", "-m", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--backend", "-b", default=None, help="Force backend: mlx or transformers")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max generation tokens")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between VLM analyses (default: 2.0)")
    args = parser.parse_args()

    # Initialize engine
    from trio_core import TrioCore, EngineConfig

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
    # Open source: camera index, video file, or RTSP URL
    source_str = args.source
    source = int(source_str) if source_str.isdigit() else source_str
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    ffmpeg_proc = None

    print(f"Ready. Opening {source_str}...")

    if is_rtsp:
        # Use system ffmpeg for RTSP (avoids macOS Local Network sandbox on OpenCV's bundled ffmpeg)
        if not shutil.which("ffmpeg"):
            print("Error: ffmpeg not found — install with: brew install ffmpeg")
            return
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", source],
            capture_output=True, text=True, timeout=10,
        )
        if not probe.stdout.strip():
            print(f"Error: Cannot probe RTSP stream: {source}")
            return
        rtsp_w, rtsp_h = [int(x) for x in probe.stdout.strip().split(",")]
        ffmpeg_proc = subprocess.Popen(
            ["ffmpeg", "-rtsp_transport", "tcp", "-i", source,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "-v", "error", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=rtsp_w * rtsp_h * 3 * 2,
        )
        cap = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot open source: {source_str}")
            print("Tip: Mac Studio needs an external webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _read_frame():
        if ffmpeg_proc:
            raw = ffmpeg_proc.stdout.read(rtsp_w * rtsp_h * 3)
            if len(raw) != rtsp_w * rtsp_h * 3:
                return False, None
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((rtsp_h, rtsp_w, 3))
        return cap.read()

    # Shared state between main thread and inference thread
    lock = threading.Lock()
    description = "Waiting for first analysis..."
    metrics_text = ""
    latest_frame_for_vlm: np.ndarray | None = None
    force_analyze = threading.Event()
    running = True

    def inference_loop():
        nonlocal description, metrics_text, latest_frame_for_vlm, running

        while running:
            # Wait for interval or force trigger
            force_analyze.wait(timeout=args.interval)
            force_analyze.clear()

            if not running:
                break

            with lock:
                frame = latest_frame_for_vlm
                if frame is None:
                    continue
                frame_copy = frame.copy()

            # Convert BGR uint8 (H,W,C) -> RGB float32 (H,W,C) for analyze_frame
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            rgb_float = rgb.astype(np.float32) / 255.0

            try:
                t0 = time.monotonic()
                result = engine.analyze_frame(rgb_float, args.prompt, max_tokens=args.max_tokens)
                elapsed = time.monotonic() - t0

                with lock:
                    description = result.text.strip().replace("\n", " ")
                    metrics_text = (
                        f"[{elapsed:.1f}s | "
                        f"preprocess {result.metrics.preprocess_ms:.0f}ms | "
                        f"inference {result.metrics.inference_ms:.0f}ms | "
                        f"{result.metrics.tokens_per_sec:.1f} tok/s]"
                    )
                print(f"\n> {description}")
                print(f"  {metrics_text}")
            except Exception as e:
                with lock:
                    description = f"Error: {e}"
                print(f"Inference error: {e}")

    # Start inference thread
    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    window_name = "TrioCore Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nPress 'q' or ESC to quit, SPACE to force re-analyze.")

    try:
        while True:
            ret, frame = _read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            # Update frame for VLM thread
            with lock:
                latest_frame_for_vlm = frame

            # Draw overlay
            with lock:
                display = draw_text_box(frame, description, metrics_text)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord(" "):  # SPACE
                force_analyze.set()
    finally:
        running = False
        force_analyze.set()  # unblock thread
        thread.join(timeout=3)
        if cap:
            cap.release()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
