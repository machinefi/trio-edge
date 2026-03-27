#!/usr/bin/env python3
"""Webcam GUI — live VLM analysis with video preview and text overlay.

Shows a window with the webcam feed and VLM description overlaid at the bottom.
Inference runs in a background thread so the video stays smooth.

Usage:
    python examples/webcam_gui.py                          # Built-in webcam (index 0)
    python examples/webcam_gui.py --source 1               # iPhone via Continuity Camera
    python examples/webcam_gui.py --source "rtsp://admin:pass@192.168.1.100:554/stream1"
    python examples/webcam_gui.py --source test_videos/intruder_house.mp4

    # Watch mode — natural language condition monitoring:
    python examples/webcam_gui.py --watch "someone is at the door"
    python examples/webcam_gui.py --watch "a package is missing" --source 1
    python examples/webcam_gui.py --watch "person not wearing safety helmet"

    # Custom prompt:
    python examples/webcam_gui.py --prompt "What is the person doing?"
    python examples/webcam_gui.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit

Camera sources:
    0           — Built-in Mac webcam
    1           — iPhone via macOS Continuity Camera (zero config, just bring iPhone near Mac)
    rtsp://...  — IP camera (Reolink, Hikvision, etc.) via RTSP
    file.mp4    — Local video file

Controls:
    q / ESC — quit
    SPACE   — force re-analyze current frame
"""

import argparse
import shutil
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np


def _wrap_text(text, font, font_scale, thickness, max_width):
    """Word-wrap text to fit within max_width pixels."""
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
    return lines


def draw_overlay(
    frame: np.ndarray,
    description: str,
    metrics: str = "",
    watch_mode: bool = False,
    triggered: bool = False,
    events: list | None = None,
    watch_text: str = "",
) -> np.ndarray:
    """Draw status bar (top) + description box (bottom) + event log."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    line_height = 22

    # --- Top status bar (watch mode only) ---
    if watch_mode:
        bar_h = 60
        color = (0, 0, 220) if triggered else (0, 160, 0)  # red or green (BGR)
        cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        overlay = frame.copy()

        status = "!! ALERT !!" if triggered else "CLEAR"
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"{status}", (margin, 42), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, ts, (w - 150, 42), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Watch condition (left side, below status bar) ---
    if watch_mode and watch_text:
        cond_y = 90
        cv2.putText(
            frame, "Watching for:", (margin, cond_y), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )  # bright cyan
        cv2.putText(
            frame, watch_text, (margin, cond_y + 25), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )  # bright green

    # --- Event log (top-right, watch mode only) — all triggered events ---
    if events:
        alerts = [e for e in events if e.get("triggered")]
        if alerts:
            log_line_h = 28
            show_alerts = alerts[-8:]
            log_w = 500
            log_x = w - log_w - margin
            log_y_start = 85 if watch_mode else 10
            for i, evt in enumerate(show_alerts):
                y = log_y_start + i * log_line_h
                cv2.putText(
                    frame, evt["text"], (log_x, y), font, 0.6, (50, 50, 255), 2, cv2.LINE_AA
                )  # bright red

    # --- Bottom description box ---
    max_width = w - 2 * margin
    lines = _wrap_text(description, font, 0.6, 1, max_width)
    if metrics:
        lines.append(metrics)

    if lines:
        box_height = len(lines) * line_height + 2 * margin
        box_top = h - box_height
        cv2.rectangle(overlay, (0, box_top), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        for i, line in enumerate(lines):
            y = box_top + margin + (i + 1) * line_height - 4
            color = (180, 180, 180) if i == len(lines) - 1 and metrics else (255, 255, 255)
            cv2.putText(frame, line, (margin, y), font, 0.6, color, 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Webcam GUI with live VLM analysis")
    parser.add_argument(
        "--source", "-s", default="0", help="RTSP URL, video file, or camera index (default: 0)"
    )
    parser.add_argument(
        "--prompt", "-p", default=None, help="Question to ask the VLM (auto-set in watch mode)"
    )
    parser.add_argument(
        "--watch",
        "-w",
        default=None,
        help="Watch condition in natural language, e.g. 'someone is at the door'",
    )
    parser.add_argument("--model", "-m", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--backend", "-b", default=None, help="Force backend: mlx or transformers")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max generation tokens")
    parser.add_argument(
        "--interval", type=float, default=3.0, help="Seconds between VLM analyses (default: 3.0)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Number of frames per analysis for temporal understanding (default: 4)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Max resolution for inference (e.g. 480, 360). Lower = faster.",
    )
    parser.add_argument("--no-sound", action="store_true", help="Disable audio alerts")
    args = parser.parse_args()

    watch_mode = args.watch is not None

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    elif watch_mode:
        prompt = (
            f"You are a security monitor. Look at this image carefully.\n"
            f'Check if the following is true: "{args.watch}"\n'
            f"{{prev_context}}"
            f"Answer YES or NO first, then briefly explain why."
        )
    else:
        prompt = "Describe what you see briefly.{prev_context}"

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
    print(f"Temporal frames: {args.frames} per analysis")
    if watch_mode:
        print(f'Watch condition: "{args.watch}"')

    # Open source
    source_str = args.source
    source = int(source_str) if source_str.isdigit() else source_str
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    ffmpeg_proc = None

    print(f"Ready. Opening {source_str}...")

    if is_rtsp:
        if not shutil.which("ffmpeg"):
            print("Error: ffmpeg not found — install with: brew install ffmpeg")
            return
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-rtsp_transport",
                "tcp",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                source,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not probe.stdout.strip():
            print(f"Error: Cannot probe RTSP stream: {source}")
            return
        rtsp_w, rtsp_h = [int(x) for x in probe.stdout.strip().split(",")]
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-i",
                source,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-an",
                "-sn",
                "-v",
                "error",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=rtsp_w * rtsp_h * 3 * 2,
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

    # --- Auto-calibrate resolution for ~1s inference ---
    if args.resolution is None:
        print("Calibrating resolution...")
        # Grab a test frame
        for _ in range(5):  # skip initial black frames
            ret, test_frame = _read_frame()
        if ret and test_frame is not None:
            h_orig, w_orig = test_frame.shape[:2]
            rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t0 = time.monotonic()
            engine.analyze_frame(rgb, "What do you see?", max_tokens=10)
            cal_time = time.monotonic() - t0
            target_time = 1.0
            if cal_time > target_time:
                scale = (target_time / cal_time) ** 0.5
                args.resolution = int(max(h_orig, w_orig) * scale)
                # Clamp to reasonable range
                args.resolution = max(args.resolution, 180)
                print(
                    f"  Full-res inference: {cal_time:.1f}s → auto-set resolution={args.resolution}px "
                    f"(~{target_time:.0f}s target)"
                )
            else:
                print(f"  Full-res inference: {cal_time:.1f}s — fast enough, no downscale needed")
        else:
            print("  Calibration failed, using full resolution")

    # Shared state
    lock = threading.Lock()
    description = "Waiting for first analysis..."
    metrics_text = ""
    latest_frame_for_vlm: np.ndarray | None = None
    frame_buffer: list[np.ndarray] = []  # recent frames for multi-frame analysis
    force_analyze = threading.Event()
    running = True
    triggered = False
    triggered_until = 0.0  # time.monotonic() when alert should expire
    events: list[dict] = []
    prev_description = ""

    def _alert(text):
        """Play audio alert on macOS."""
        if args.no_sound:
            return
        try:
            subprocess.Popen(
                ["say", "-r", "200", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            pass

    def inference_loop():
        nonlocal description, metrics_text, latest_frame_for_vlm, running
        nonlocal triggered, triggered_until, prev_description

        while running:
            force_analyze.wait(timeout=args.interval)
            force_analyze.clear()

            if not running:
                break

            # Grab recent frames for temporal understanding
            with lock:
                if not frame_buffer:
                    continue
                # Take up to N_FRAMES evenly spaced from buffer
                buf = list(frame_buffer)

            n_frames = args.frames
            if len(buf) >= n_frames:
                indices = np.linspace(0, len(buf) - 1, n_frames, dtype=int)
                selected = [buf[i] for i in indices]
            else:
                selected = buf

            # Resize + convert BGR uint8 -> RGB float32, stack as (N, C, H, W)
            rgb_frames = []
            for f in selected:
                # Downscale for faster inference
                if args.resolution:
                    rh, rw = f.shape[:2]
                    scale = args.resolution / max(rh, rw)
                    if scale < 1.0:
                        f = cv2.resize(f, (int(rw * scale), int(rh * scale)))
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rgb_frames.append(rgb.transpose(2, 0, 1))  # HWC -> CHW
            video_array = np.stack(rgb_frames)  # (N, C, H, W)

            # Build prompt with temporal context
            prev_ctx = ""
            if prev_description:
                prev_ctx = f'\nPrevious observation: "{prev_description}"\n'
            current_prompt = prompt.replace("{prev_context}", prev_ctx)

            try:
                t0 = time.monotonic()
                result = engine.analyze_video(
                    video_array, current_prompt, max_tokens=args.max_tokens
                )
                elapsed = time.monotonic() - t0
                text = result.text.strip().replace("\n", " ")

                with lock:
                    description = text
                    prev_description = text
                    metrics_text = (
                        f"[{elapsed:.1f}s | "
                        f"preprocess {result.metrics.preprocess_ms:.0f}ms | "
                        f"inference {result.metrics.inference_ms:.0f}ms | "
                        f"{result.metrics.tokens_per_sec:.1f} tok/s]"
                    )

                    # Watch mode: check if triggered
                    if watch_mode:
                        upper = text.upper()
                        is_alert = upper.startswith("YES")
                        if is_alert:
                            was_triggered = triggered
                            triggered = True
                            triggered_until = time.monotonic() + 2.0
                            ts = datetime.now().strftime("%H:%M:%S")
                            reason = text[3:].lstrip(".,!: ") if len(text) > 3 else text
                            events.append(
                                {
                                    "text": f"[{ts}] {reason[:55]}",
                                    "triggered": True,
                                }
                            )
                            # Only speak on state change (CLEAR -> ALERT)
                            if not was_triggered:
                                _alert(f"Alert: {args.watch}")

                print(f"\n> {text}")
                print(f"  {metrics_text}")
            except Exception as e:
                with lock:
                    description = f"Error: {e}"
                print(f"Inference error: {e}")

    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    window_name = "TrioCore" + (" Watch" if watch_mode else " Webcam")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    controls = "Press 'q' or ESC to quit, SPACE to force re-analyze."
    print(f"\n{controls}")

    try:
        while True:
            ret, frame = _read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            with lock:
                latest_frame_for_vlm = frame
                frame_buffer.append(frame.copy())
                # Keep buffer bounded
                max_buf = args.frames * 3
                if len(frame_buffer) > max_buf:
                    frame_buffer[:] = frame_buffer[-max_buf:]

            with lock:
                # Auto-reset alert after 3 seconds
                if triggered and time.monotonic() > triggered_until:
                    triggered = False

                display = draw_overlay(
                    frame,
                    description,
                    metrics_text,
                    watch_mode=watch_mode,
                    triggered=triggered,
                    events=events if watch_mode else None,
                    watch_text=args.watch or "",
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                force_analyze.set()
    finally:
        running = False
        force_analyze.set()
        thread.join(timeout=3)
        if cap:
            cap.release()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        cv2.destroyAllWindows()

        if events:
            print(f"\n--- Session Summary: {len(events)} checks ---")
            alerts = [e for e in events if e["triggered"]]
            print(f"Alerts: {len(alerts)}/{len(events)}")
            for e in alerts:
                print(f"  {e['text']}")
        print("Done.")


if __name__ == "__main__":
    main()
