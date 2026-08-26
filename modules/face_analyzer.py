"""
modules/face_analyzer.py
────────────────────────
Handles:
  • Webcam frame capture
  • Face detection
  • Age & gender estimation via DeepFace
  • Annotating frames with results
"""

import cv2
import threading
import numpy as np
from deepface import DeepFace

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FaceAnalyzer:
    """
    Runs in a background thread.
    Continuously reads webcam frames, analyzes every Nth frame
    for age/gender using DeepFace, and exposes the latest
    annotated frame + analysis results.
    """

    def __init__(self):
        self._cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Latest state (thread-safe access via lock)
        self.latest_frame = None          # BGR numpy array
        self.latest_results = []          # list of dicts: {age, gender, box}
        self._frame_count = 0

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self):
        """Open the webcam and start the analysis thread."""
        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Check WEBCAM_INDEX in config.py")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to stop and release the webcam."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def get_latest(self):
        """
        Returns (annotated_frame, results_list) — thread-safe.
        annotated_frame : BGR numpy array or None
        results_list    : list of dicts [{age, gender, box}, ...]
        """
        with self._lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            results = list(self.latest_results)
        return frame, results

    # ── Internal Loop ───────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            # Run DeepFace only every N frames to save CPU/battery
            if self._frame_count % config.FACE_ANALYSIS_EVERY_N_FRAMES == 0:
                results = self._analyze(frame)
            else:
                with self._lock:
                    results = list(self.latest_results)

            annotated = self._draw(frame, results)

            with self._lock:
                self.latest_frame = annotated
                self.latest_results = results

    def _analyze(self, frame):
        """
        Run DeepFace analysis using OpenCV backend (no extra model download).
        Falls back to Haar cascade if DeepFace fails.
        """
        try:
            analyses = DeepFace.analyze(
                img_path=frame,
                actions=["age", "gender"],
                detector_backend="opencv",   # fast, built-in, no extra download
                enforce_detection=False,
                silent=True,
            )
            # DeepFace returns a list or a single dict — normalise to list
            if isinstance(analyses, dict):
                analyses = [analyses]

            results = []
            for a in analyses:
                region = a.get("region", {})
                w = region.get("w", 0)
                h = region.get("h", 0)
                # skip garbage zero-size detections
                if w < 20 or h < 20:
                    continue
                results.append({
                    "age":    int(a.get("age", 0)),
                    "gender": a.get("dominant_gender", "Unknown"),
                    "box":    (
                        region.get("x", 0),
                        region.get("y", 0),
                        w, h,
                    ),
                })
            return results

        except Exception as e:
            # Silent fallback — just return empty so UI shows "No face detected"
            return []

    def _draw(self, frame, results):
        """Draw bounding boxes and labels on a copy of the frame."""
        out = frame.copy()
        for r in results:
            x, y, w, h = r["box"]
            age    = r["age"]
            gender = r["gender"]

            color = (0, 200, 0) if gender.lower() == "man" else (200, 0, 200)

            # Bounding box
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            # Label background + text
            label = f"{gender}, ~{age} yrs"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y - th - 10), (x + tw + 4, y), color, -1)
            cv2.putText(
                out, label,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2,
            )
        return out
