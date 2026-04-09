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

    # Digest mode — smart event timeline:
    python examples/webcam_gui.py --digest

    # Count mode — cumulative object counting:
    python examples/webcam_gui.py --count

    # Custom prompt:
    python examples/webcam_gui.py --prompt "What is the person doing?"

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
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np


@dataclass
class WebcamGUIConfig:
    """Structured config accepted by main() when called programmatically."""

    source: str = "0"
    prompt: str | None = None
    watch: str | None = None
    model: str | None = None
    backend: str | None = None
    max_tokens: int = 80
    interval: float = 3.0
    frames: int = 4
    resolution: int | None = None
    no_sound: bool = False
    count: bool = False
    digest: bool = False
    adapter: str | None = None


def _parse_counts(text):
    """Parse 'COUNT people:N cars:N dogs:N cats:N' from VLM response."""
    counts = {}
    for key in ("people", "cars", "dogs", "cats"):
        m = re.search(rf"{key}:\s*(\d+)", text, re.IGNORECASE)
        counts[key] = int(m.group(1)) if m else 0
    return counts


def _text_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Check if two event descriptions are similar (word overlap ratio)."""
    if not a or not b:
        return False
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b)) >= threshold


def _parse_digest(text):
    """Parse digest response: 'EVENT: ...' or 'NOTHING' from VLM."""
    text = re.sub(r"</?think>", "", text).strip()
    # Check for "nothing" / "no change" variants
    lower = text.lower()
    if (
        lower.startswith("nothing")
        or lower.startswith("no change")
        or lower.startswith("no new")
        or lower.startswith("no,")
        or "there is no person" in lower
        or "no person, animal" in lower
        or "there are no" in lower
    ):
        return None
    # Strip "EVENT:" prefix if present
    text = re.sub(r"^EVENT:\s*", "", text, flags=re.IGNORECASE).strip()
    return text if text else None


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
    counters: dict | None = None,
    digest_events: list | None = None,
    chat_input: str = "",
    chat_history: list | None = None,
) -> np.ndarray:
    """Draw status bar (top) + description box (bottom) + event log + counters."""
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
            # Dark background behind event log
            log_h = len(show_alerts) * log_line_h + 10
            cv2.rectangle(
                overlay, (log_x - 5, log_y_start - 20), (w, log_y_start + log_h - 10), (0, 0, 0), -1
            )
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            overlay = frame.copy()
            for i, evt in enumerate(show_alerts):
                y = log_y_start + i * log_line_h
                cv2.putText(
                    frame, evt["text"], (log_x, y), font, 0.6, (100, 100, 255), 2, cv2.LINE_AA
                )

    # --- Digest event timeline (left side, scrolling) ---
    if digest_events:
        panel_y = 10
        log_line_h = 26
        # Only show events that haven't expired (5s per event, newest lives longest)
        now = time.monotonic()
        visible = [e for e in digest_events if e.get("expire", now + 1) > now]
        show_events = visible[-8:]  # max 8 on screen
        panel_h = len(show_events) * log_line_h + 20
        panel_w = min(w - 20, 650)
        # Dark semi-transparent background
        cv2.rectangle(overlay, (0, panel_y - 5), (panel_w, panel_y + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        overlay = frame.copy()

        cv2.putText(
            frame, "Event Log", (margin, panel_y + 14), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA
        )
        for i, evt in enumerate(show_events):
            y = panel_y + 35 + i * log_line_h
            text = evt["text"][:80]
            # Color: cyan for timestamp, white for text
            cv2.putText(
                frame, text, (margin + 2, y), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA
            )  # shadow
            cv2.putText(frame, text, (margin, y), font, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    # --- Counters panel (left side, with dark background) ---
    if counters:
        panel_y = 130 if watch_mode else 10
        panel_h = 30 + 4 * 32 + 10  # title + 4 rows + padding
        panel_w = 200
        # Dark semi-transparent background
        cv2.rectangle(overlay, (0, panel_y - 10), (panel_w, panel_y + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        overlay = frame.copy()

        cv2.putText(
            frame,
            "Cumulative Count",
            (margin, panel_y + 5),
            font,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        labels = {"people": "People", "cars": "Cars", "dogs": "Dogs", "cats": "Cats"}
        for i, key in enumerate(["people", "cars", "dogs", "cats"]):
            y = panel_y + 35 + i * 32
            total = counters.get(key, 0)
            label = f"{labels[key]}: {total}"
            # White text with black outline for readability
            cv2.putText(
                frame, label, (margin + 5, y), font, 0.75, (0, 0, 0), 3, cv2.LINE_AA
            )  # black outline
            cv2.putText(
                frame, label, (margin + 5, y), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA
            )  # white fill

    # --- Chat history + input bar (bottom) ---
    bottom_lines = []

    # Chat history (last 3 Q&A pairs)
    if chat_history:
        for entry in chat_history[-3:]:
            bottom_lines.append(("Q: " + entry["q"], (0, 255, 255)))  # cyan
            for al in _wrap_text("A: " + entry["a"], font, 0.55, 1, w - 2 * margin):
                bottom_lines.append((al, (200, 255, 200)))  # green

    # Current description + metrics (if no chat active)
    if not chat_input and not chat_history:
        for line in _wrap_text(description, font, 0.6, 1, w - 2 * margin):
            bottom_lines.append((line, (255, 255, 255)))
        if metrics:
            bottom_lines.append((metrics, (180, 180, 180)))
    elif not chat_input:
        # Show metrics line even with chat
        if metrics:
            bottom_lines.append((metrics, (180, 180, 180)))

    # Chat input line
    input_line = f"> {chat_input}_" if chat_input is not None else ""
    if input_line:
        bottom_lines.append((input_line, (255, 255, 0)))  # yellow

    if bottom_lines:
        box_height = len(bottom_lines) * line_height + 2 * margin
        box_top = h - box_height
        cv2.rectangle(overlay, (0, box_top), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        for i, (line, color) in enumerate(bottom_lines):
            y = box_top + margin + (i + 1) * line_height - 4
            cv2.putText(frame, line, (margin, y), font, 0.55, color, 1, cv2.LINE_AA)

    return frame


def main(config: WebcamGUIConfig | None = None):
    if config is not None:
        args = config
    else:
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
        parser.add_argument(
            "--model", "-m", default=None, help="Model name (auto-detected if omitted)"
        )
        parser.add_argument(
            "--backend", "-b", default=None, help="Force backend: mlx or transformers"
        )
        parser.add_argument("--max-tokens", type=int, default=80, help="Max generation tokens")
        parser.add_argument(
            "--interval",
            type=float,
            default=3.0,
            help="Seconds between VLM analyses (default: 3.0)",
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
        parser.add_argument(
            "--count", action="store_true", help="Count people, cars, and dogs (cumulative)"
        )
        parser.add_argument(
            "--digest",
            action="store_true",
            help="Smart event timeline — logs activities with scene understanding",
        )
        parser.add_argument("--adapter", default=None, help="LoRA adapter directory path")
        args = parser.parse_args()

    watch_mode = args.watch is not None
    count_mode = args.count
    digest_mode = args.digest

    count_suffix = (
        (
            "\nAlso carefully count ALL visible objects including small or partially occluded ones. "
            "Look closely for animals (dogs, cats) even if they are small, far away, or partially hidden. "
            "End your answer with a line in EXACTLY this format:\n"
            "COUNT people:N cars:N dogs:N cats:N"
        )
        if count_mode
        else ""
    )

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    elif digest_mode:
        prompt = (
            "Something moved in this security camera scene. "
            "Identify ONLY the moving subject (person, animal, or vehicle in motion). "
            "IGNORE all parked/stationary vehicles — they are not the cause of motion. "
            "Describe the moving subject in one sentence: appearance, action, direction."
        )
    elif count_mode and not watch_mode:
        prompt = (
            "You are a security camera counter tracking objects over time.\n"
            "Count every person, car, dog, and cat CURRENTLY visible — even small, distant, or partially hidden.\n"
            "{prev_context}"
            "Report only what is visible RIGHT NOW (not cumulative). "
            "Briefly describe the scene, then end with EXACTLY:\n"
            "COUNT people:N cars:N dogs:N cats:N"
        )
    elif watch_mode:
        prompt = (
            f"You are a security monitor. Look at this image carefully.\n"
            f'Check if the following is true: "{args.watch}"\n'
            f"{{prev_context}}"
            f"Answer YES or NO first, then briefly explain why."
            f"{count_suffix}"
        )
    else:
        prompt = "Describe what you see briefly.{prev_context}" + count_suffix

    # Initialize engine
    from trio_core import EngineConfig, TrioCore

    config_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        config_kwargs["model"] = args.model
    if args.adapter:
        config_kwargs["adapter_path"] = args.adapter
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
    if digest_mode:
        print("Mode: Event digest (smart activity timeline)")

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
        # Auto-proxy for Tailscale on macOS
        from trio_core._rtsp_proxy import ensure_rtsp_url

        proxied = ensure_rtsp_url(source)
        if proxied != source:
            time.sleep(1)  # let proxy settle
            source = proxied
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
            timeout=20,
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
    frame_buffer: list[np.ndarray] = []  # recent frames for multi-frame analysis
    force_analyze = threading.Event()
    running = True
    triggered = False
    triggered_until = 0.0  # time.monotonic() when alert should expire
    events: list[dict] = []
    prev_description = ""
    # Counting state — debounce prevents flicker double-counts
    total_counts = {"people": 0, "cars": 0, "dogs": 0, "cats": 0}
    prev_visible = {"people": 0, "cars": 0, "dogs": 0, "cats": 0}
    zero_streak = {"people": 0, "cars": 0, "dogs": 0, "cats": 0}  # consecutive 0-count frames
    DEBOUNCE_FRAMES = 3  # require N consecutive zeros before accepting decrease
    # Digest state
    digest_events: list[dict] = []
    nothing_count = 0  # consecutive "nothing" responses
    # Chat state
    chat_input = ""
    chat_history: list[dict] = []  # [{"q": "...", "a": "..."}, ...]
    chat_busy = False
    # Motion detection baseline (for digest mode)
    baseline_frame: np.ndarray | None = None
    MOTION_THRESHOLD = 0.005  # fraction of pixels that must change
    motion_mask: np.ndarray | None = None  # binary mask of moving regions

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
        nonlocal description, metrics_text, running
        nonlocal triggered, triggered_until, prev_description
        nonlocal total_counts, prev_visible, nothing_count
        nonlocal baseline_frame, motion_mask

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

            # Motion gate for digest mode — skip VLM if scene is static
            if digest_mode:
                current_gray = cv2.cvtColor(selected[-1], cv2.COLOR_BGR2GRAY)
                current_gray = cv2.GaussianBlur(current_gray, (11, 11), 0)
                if baseline_frame is None:
                    baseline_frame = current_gray
                    with lock:
                        description = "Monitoring... no motion."
                    continue
                diff = cv2.absdiff(baseline_frame, current_gray)
                motion_pixels = np.sum(diff > 15) / diff.size
                # Update baseline slowly (rolling average)
                baseline_frame = cv2.addWeighted(
                    baseline_frame, 0.95, current_gray, 0.05, 0
                ).astype(np.uint8)
                if motion_pixels < MOTION_THRESHOLD:
                    nothing_count += 1
                    motion_mask = None
                    if nothing_count % 100 == 1:
                        print(f"  ... no motion ({nothing_count}) score={motion_pixels:.4f}")
                    with lock:
                        description = f"Monitoring... (motion: {motion_pixels:.4f})"
                    continue
                # Build motion mask — dilate to cover full moving objects
                raw_mask = (diff > 15).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                motion_mask = cv2.dilate(raw_mask, kernel, iterations=2)
                motion_mask = cv2.GaussianBlur(motion_mask, (41, 41), 0)
                print(f"  [MOTION DETECTED: {motion_pixels:.4f}]")

            # Resize + convert BGR uint8 -> RGB float32, stack as (N, C, H, W)
            # In digest mode, dim static regions so VLM focuses on motion
            rgb_frames = []
            for f in selected:
                # Apply motion highlight: darken static areas
                if digest_mode and motion_mask is not None:
                    fh, fw = f.shape[:2]
                    mh, mw = motion_mask.shape[:2]
                    if (fh, fw) != (mh, mw):
                        mask_resized = cv2.resize(motion_mask, (fw, fh))
                    else:
                        mask_resized = motion_mask
                    # Blend: moving regions stay bright, static dims to 30%
                    alpha = mask_resized.astype(np.float32) / 255.0
                    alpha = alpha[:, :, np.newaxis]  # (H,W,1) for broadcasting
                    f = (f * (0.3 + 0.7 * alpha)).astype(np.uint8)
                # Downscale for faster inference
                if args.resolution:
                    rh, rw = f.shape[:2]
                    scale = args.resolution / max(rh, rw)
                    if scale < 1.0:
                        f = cv2.resize(f, (int(rw * scale), int(rh * scale)))
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rgb_frames.append(rgb.transpose(2, 0, 1))  # HWC -> CHW
            video_array = np.stack(rgb_frames)  # (N, C, H, W)

            # Build prompt — digest mode skips prev_context to avoid hallucination feedback
            if digest_mode:
                current_prompt = prompt
            else:
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
                # Strip thinking tags
                answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                answer = re.sub(r"</?think>", "", answer).strip()
                # Truncate at chat template leakage (model didn't stop at EOS)
                answer = re.sub(r"\s*user\b.*", "", answer, flags=re.IGNORECASE).strip()
                answer = re.sub(r"\s*assistant\b.*", "", answer, flags=re.IGNORECASE).strip()
                answer = re.sub(r"<\|.*", "", answer).strip()  # <|im_end|> etc

                # Update cumulative counters with debounce
                if count_mode:
                    current = _parse_counts(answer)
                    for key in total_counts:
                        if current[key] > 0:
                            zero_streak[key] = 0
                            if current[key] > prev_visible[key]:
                                total_counts[key] += current[key] - prev_visible[key]
                                prev_visible[key] = current[key]
                            # On decrease: keep prev_visible as high-water mark
                            # so fluctuations (3→2→4) don't over-count
                        else:
                            # Only accept decrease to 0 after N consecutive zero frames
                            zero_streak[key] += 1
                            if zero_streak[key] >= DEBOUNCE_FRAMES:
                                prev_visible[key] = 0
                    # Strip COUNT line from display text
                    answer = re.sub(
                        r"\s*COUNT\s+people:\d+\s+cars:\d+\s+dogs:\d+\s+cats:\d+", "", answer
                    ).strip()

                # Digest mode: motion gate already confirmed activity, log it
                if digest_mode:
                    nothing_count = 0
                    ts = datetime.now().strftime("%H:%M:%S")
                    event_text = answer.strip()
                    if event_text:
                        # Dedup: skip if too similar to last event
                        last_raw = digest_events[-1]["raw"] if digest_events else ""
                        if _text_similar(event_text, last_raw):
                            # Same event continuing — refresh expiry
                            digest_events[-1]["time"] = ts
                            digest_events[-1]["text"] = f"[{ts}] {event_text}"
                            digest_events[-1]["expire"] = time.monotonic() + 8.0
                        else:
                            digest_events.append(
                                {
                                    "time": ts,
                                    "text": f"[{ts}] {event_text}",
                                    "raw": event_text,
                                    "expire": time.monotonic() + 8.0,
                                }
                            )
                            print(f"  [{ts}] {event_text}")
                        # Prune expired events to prevent unbounded growth
                        now = time.monotonic()
                        digest_events[:] = [
                            e for e in digest_events if e.get("expire", now + 1) > now - 60
                        ]

                with lock:
                    description = answer
                    # For digest: feed back clean text to avoid garbage feedback loop
                    if digest_mode:
                        event_text_clean = _parse_digest(answer)
                        prev_description = (
                            event_text_clean if event_text_clean else "quiet street, no activity"
                        )
                    else:
                        prev_description = answer
                    metrics_text = (
                        f"[{elapsed:.1f}s | "
                        f"preprocess {result.metrics.preprocess_ms:.0f}ms | "
                        f"inference {result.metrics.inference_ms:.0f}ms | "
                        f"{result.metrics.tokens_per_sec:.1f} tok/s]"
                    )

                    # Watch mode: check if triggered
                    if watch_mode:
                        upper = answer.upper()
                        is_alert = upper.startswith("YES")
                        if is_alert:
                            was_triggered = triggered
                            triggered = True
                            triggered_until = time.monotonic() + 2.0
                            ts = datetime.now().strftime("%H:%M:%S")
                            reason = answer[3:].lstrip(".,!: ") if len(answer) > 3 else answer
                            events.append(
                                {
                                    "text": f"[{ts}] {reason[:55]}",
                                    "triggered": True,
                                }
                            )
                            # Only speak on state change (CLEAR -> ALERT)
                            if not was_triggered:
                                _alert(f"Alert: {args.watch}")

                if not digest_mode:
                    print(f"\n> {answer}")
                    print(f"  {metrics_text}")
            except Exception as e:
                with lock:
                    description = f"Error: {e}"
                print(f"Inference error: {e}")

    def chat_query(question: str):
        """Send a question to VLM with current frame, update chat_history."""
        nonlocal chat_busy
        chat_busy = True
        with lock:
            buf = list(frame_buffer)
        if not buf:
            chat_history.append({"q": question, "a": "No frame available."})
            chat_busy = False
            return
        frame_for_chat = buf[-1]
        if args.resolution:
            rh, rw = frame_for_chat.shape[:2]
            scale = args.resolution / max(rh, rw)
            if scale < 1.0:
                frame_for_chat = cv2.resize(frame_for_chat, (int(rw * scale), int(rh * scale)))
        rgb = cv2.cvtColor(frame_for_chat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            result = engine.analyze_frame(rgb, question, max_tokens=80)
            answer = re.sub(r"<think>.*?</think>", "", result.text.strip(), flags=re.DOTALL).strip()
            answer = re.sub(r"</?think>", "", answer).strip()
            answer = re.sub(r"\s*user\b.*", "", answer, flags=re.IGNORECASE).strip()
            answer = re.sub(r"<\|.*", "", answer).strip()
            chat_history.append({"q": question, "a": answer or "No answer."})
            print(f"  Chat Q: {question}")
            print(f"  Chat A: {answer}")
        except Exception as e:
            chat_history.append({"q": question, "a": f"Error: {e}"})
        chat_busy = False

    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    mode_label = " Watch" if watch_mode else " Digest" if digest_mode else " Webcam"
    window_name = "TrioCore" + mode_label
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    controls = "Type to ask questions, Enter to send, ESC to clear/quit, SPACE to re-analyze."
    print(f"\n{controls}")

    try:
        while True:
            ret, frame = _read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            with lock:
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
                    counters=total_counts if count_mode else None,
                    digest_events=digest_events if digest_mode else None,
                    chat_input=chat_input,
                    chat_history=chat_history,
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # no key pressed
                pass
            elif key == 27:  # ESC — clear chat input or quit
                if chat_input:
                    chat_input = ""
                else:
                    break
            elif key == ord("q") and not chat_input:
                break
            elif key == 13:  # Enter — submit chat question
                if chat_input.strip() and not chat_busy:
                    q = chat_input.strip()
                    chat_input = ""
                    threading.Thread(target=chat_query, args=(q,), daemon=True).start()
            elif key == 8 or key == 127:  # Backspace
                chat_input = chat_input[:-1]
            elif key == ord(" ") and not chat_input:
                force_analyze.set()
            elif 32 <= key < 127:  # printable ASCII
                chat_input += chr(key)
    finally:
        running = False
        force_analyze.set()
        thread.join(timeout=3)
        if cap:
            cap.release()
        if ffmpeg_proc:
            # Close pipes first so ffmpeg doesn't block on a full stdout buffer
            # (the RTSP raw-video pipe produces ~27 MB/s, and once the read loop
            # stops the OS pipe buffer fills in <1 ms, blocking ffmpeg's write and
            # preventing it from handling SIGTERM).
            if ffmpeg_proc.stdout:
                ffmpeg_proc.stdout.close()
            if ffmpeg_proc.stderr:
                ffmpeg_proc.stderr.close()
            ffmpeg_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait()
        cv2.destroyAllWindows()

        if events:
            print(f"\n--- Session Summary: {len(events)} checks ---")
            alerts = [e for e in events if e["triggered"]]
            print(f"Alerts: {len(alerts)}/{len(events)}")
            for e in alerts:
                print(f"  {e['text']}")
        if count_mode and any(total_counts.values()):
            print("\n--- Object Counts ---")
            for key, val in total_counts.items():
                if val > 0:
                    print(f"  {key}: {val}")
        if digest_mode and digest_events:
            print(f"\n{'=' * 60}")
            print(f"  ACTIVITY DIGEST — {len(digest_events)} events")
            print(f"{'=' * 60}")
            for evt in digest_events:
                print(f"  {evt['text']}")
            print(f"{'=' * 60}")
        print("Done.")


if __name__ == "__main__":
    main()
