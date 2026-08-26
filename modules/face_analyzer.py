"""
modules/face_analyzer.py
────────────────────────
Improvements in v2:
  • Tries mediapipe backend first (faster, angle-robust), falls back to opencv
  • Confidence threshold filtering — ignores low-confidence detections
  • Temporal smoothing — results are blended across frames (no flickering)
  • Better drawing — shows confidence %, color-coded by gender
"""

import cv2
import threading
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# DeepFace import — graceful if not installed
try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
except ImportError:
    _DEEPFACE_OK = False
    print("[FaceAnalyzer] DeepFace not found — age/gender disabled.")

# Detect best available backend at import time
def _best_backend():
    try:
        import mediapipe  # noqa
        return "mediapipe"
    except ImportError:
        return "opencv"

_BACKEND = _best_backend()
print(f"[FaceAnalyzer] Using detector backend: {_BACKEND}")


class FaceAnalyzer:
    """
    Threaded face analyzer.

    Each webcam frame is captured and every Nth frame is analyzed by DeepFace
    for age and gender. Results are smoothed across frames.

    Public API:
        start()        — open webcam + start thread
        stop()         — stop thread + release camera
        get_latest()   — returns (annotated_frame, results_list)
    """

    # Minimum confidence (0–100) to accept an age/gender prediction
    CONFIDENCE_THRESHOLD = config.FACE_CONFIDENCE_THRESHOLD

    def __init__(self):
        self._cap        = None
        self._thread     = None
        self._running    = False
        self._lock       = threading.Lock()
        self._frame_count = 0

        # Latest state (thread-safe)
        self.latest_frame   = None   # BGR ndarray
        self.latest_results = []     # list of result dicts
        self._prev_results  = []     # for temporal smoothing

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self):
        """Open webcam and start background analysis thread."""
        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check WEBCAM_INDEX in config.py"
            )
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def get_latest(self):
        """Thread-safe read of the latest annotated frame and results."""
        with self._lock:
            frame   = self.latest_frame.copy() if self.latest_frame is not None else None
            results = list(self.latest_results)
        return frame, results

    # ── Internal Loop ───────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            # Analyze every N frames to save battery
            if self._frame_count % config.FACE_ANALYSIS_EVERY_N_FRAMES == 0:
                new_results = self._analyze(frame)
                # Smooth: if we got results use them, otherwise keep previous
                results = new_results if new_results else self._prev_results
                self._prev_results = results
            else:
                with self._lock:
                    results = list(self.latest_results)

            annotated = self._draw(frame, results)

            with self._lock:
                self.latest_frame   = annotated
                self.latest_results = results

    # ── Analysis ────────────────────────────────────────────────────────────────

    def _analyze(self, frame):
        """
        Run DeepFace age/gender analysis.
        Tries mediapipe backend, falls back to opencv.
        Filters results by confidence threshold.
        """
        if not _DEEPFACE_OK:
            return []

        for backend in [_BACKEND, "opencv"]:
            try:
                analyses = DeepFace.analyze(
                    img_path=frame,
                    actions=["age", "gender"],
                    detector_backend=backend,
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(analyses, dict):
                    analyses = [analyses]

                results = []
                for a in analyses:
                    region = a.get("region", {})
                    w = region.get("w", 0)
                    h = region.get("h", 0)

                    # Skip tiny/garbage detections
                    if w < 30 or h < 30:
                        continue

                    # Gender confidence — handle key case mismatch between
                    # dominant_gender ("Man") and gender dict keys ("man")
                    gender_scores   = a.get("gender", {})
                    dominant_gender = a.get("dominant_gender", "Unknown")
                    confidence      = 0.0
                    if isinstance(gender_scores, dict):
                        if dominant_gender in gender_scores:
                            confidence = gender_scores[dominant_gender]
                        else:
                            # case-insensitive fallback
                            for k, v in gender_scores.items():
                                if k.lower() == dominant_gender.lower():
                                    confidence = v
                                    break
                            else:
                                # take the maximum score available
                                confidence = max(gender_scores.values()) if gender_scores else 100
                    else:
                        confidence = 100  # no score dict — trust the result

                    # Uncomment the line below to debug detection issues:
                    # print(f"[Face] {dominant_gender} conf={confidence:.1f}% box=({w}x{h})")

                    if confidence < self.CONFIDENCE_THRESHOLD:
                        continue

                    results.append({
                        "age":        int(a.get("age", 0)),
                        "gender":     dominant_gender,
                        "confidence": round(confidence, 1),
                        "box":        (
                            region.get("x", 0),
                            region.get("y", 0),
                            w, h,
                        ),
                    })

                if results:
                    return results   # return on first successful backend
            except Exception:
                continue  # try next backend

        return []

    # ── Drawing ─────────────────────────────────────────────────────────────────

    def _draw(self, frame, results):
        """Draw bounding boxes, gender labels, age, and confidence on frame."""
        out = frame.copy()

        for r in results:
            x, y, w, h   = r["box"]
            age           = r["age"]
            gender        = r["gender"]
            conf          = r.get("confidence", 0)

            # Color: green for Man, magenta for Woman
            is_man = "man" in gender.lower()
            color  = (0, 210, 0) if is_man else (210, 0, 210)

            # Fancy bounding box with corner markers
            _draw_fancy_box(out, x, y, w, h, color)

            # Label pill
            label = f"{'Male' if is_man else 'Female'}  ~{age}y  {conf:.0f}%"
            _draw_label(out, label, x, y, color)

        return out


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_fancy_box(img, x, y, w, h, color, thickness=2, corner_len=20):
    """Draw a bounding box with corner accent marks."""
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, 1)  # thin full rect
    # Corners
    for cx, cy, dx, dy in [
        (x,  y,  1,  1), (x2, y,  -1,  1),
        (x,  y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, thickness)


def _draw_label(img, text, x, y, color):
    """Draw a filled pill label above the bounding box."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    y0  = max(y - th - pad * 2, 0)
    cv2.rectangle(img, (x, y0), (x + tw + pad * 2, y0 + th + pad * 2), color, -1)
    cv2.putText(img, text, (x + pad, y0 + th + pad - 1), font, scale, (255, 255, 255), thickness)
