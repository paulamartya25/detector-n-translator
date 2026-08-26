"""
modules/face_analyzer.py  v4
────────────────────────────
Robust OpenCV DNN face analysis — no DeepFace, no mediapipe, works on Python 3.13.

Tiers:
  Tier 1 — SSD ResNet face detector  (already downloaded ~10 MB)
            + Levi & Hassner age/gender CNNs (~37 MB each, auto-download)
  Tier 2 — SSD ResNet face detector only (if age/gender models fail)
            shows bounding box + "Face detected" label
  Tier 3 — Haar Cascade (zero download, built into OpenCV)
            always works as last resort
"""

import cv2
import os
import threading
import urllib.request
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

_FACE_PROTO   = os.path.join(_MODELS_DIR, "deploy.prototxt")
_FACE_MODEL   = os.path.join(_MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
_AGE_PROTO    = os.path.join(_MODELS_DIR, "age_deploy.prototxt")
_AGE_MODEL    = os.path.join(_MODELS_DIR, "age_net.caffemodel")
_GENDER_PROTO = os.path.join(_MODELS_DIR, "gender_deploy.prototxt")
_GENDER_MODEL = os.path.join(_MODELS_DIR, "gender_net.caffemodel")

# ── Download URLs (multiple mirrors per file) ─────────────────────────────────
_FACE_URLS = {
    _FACE_PROTO: [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    ],
    _FACE_MODEL: [
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    ],
}

# Age/gender models — spmallick's learnopencv mirror (smaller, ~37 MB each)
_AG_URLS = {
    _AGE_PROTO: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/age_deploy.prototxt",
    ],
    _AGE_MODEL: [
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_gender_models/age_net.caffemodel",
    ],
    _GENDER_PROTO: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/gender_deploy.prototxt",
    ],
    _GENDER_MODEL: [
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_gender_models/gender_net.caffemodel",
    ],
}

AGE_BUCKETS   = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                 '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']
_MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)


# ── Download helpers ───────────────────────────────────────────────────────────

def _download(path: str, urls: list) -> bool:
    """Try each URL in order; return True if file obtained."""
    fname = os.path.basename(path)
    for url in urls:
        try:
            print(f"[FaceAnalyzer] Downloading {fname}…")
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1_048_576
            print(f"[FaceAnalyzer] ✓ {fname} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"[FaceAnalyzer] ✗ {url} failed: {e}")
            if os.path.exists(path):          # remove partial download
                os.remove(path)
    return False


def _ensure_face_models() -> bool:
    """Download SSD face detector models. Returns True if ready."""
    for path, urls in _FACE_URLS.items():
        if not os.path.exists(path):
            if not _download(path, urls):
                return False
    return True


def _ensure_ag_models() -> bool:
    """Download age/gender models in background. Returns True if all ready."""
    for path, urls in _AG_URLS.items():
        if not os.path.exists(path):
            if not _download(path, urls):
                return False
    return True


# ── Analyzer class ────────────────────────────────────────────────────────────

class FaceAnalyzer:
    """
    Threaded face analyzer — pure OpenCV DNN, no external deep learning deps.

    Degrades gracefully:
      SSD + Age/Gender  →  SSD only  →  Haar cascade
    """

    def __init__(self):
        self._cap         = None
        self._thread      = None
        self._running     = False
        self._lock        = threading.Lock()
        self._frame_count = 0

        # Model handles
        self._face_net    = None
        self._age_net     = None
        self._gender_net  = None

        # Result state
        self.latest_frame   = None
        self.latest_results = []
        self._prev_results  = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check WEBCAM_INDEX in config.py")

        # Load face detector immediately (already downloaded)
        if _ensure_face_models():
            try:
                self._face_net = cv2.dnn.readNet(_FACE_MODEL, _FACE_PROTO)
                print("[FaceAnalyzer] ✓ SSD face detector ready")
            except Exception as e:
                print(f"[FaceAnalyzer] SSD load failed ({e}) — using Haar cascade")

        # Download age/gender models in background (non-blocking)
        threading.Thread(target=self._load_ag_models, daemon=True).start()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def get_latest(self):
        with self._lock:
            frame   = self.latest_frame.copy() if self.latest_frame is not None else None
            results = list(self.latest_results)
        return frame, results

    # ── Age/Gender model loader (runs in background) ───────────────────────────

    def _load_ag_models(self):
        """Download and load age/gender models without blocking the UI."""
        if _ensure_ag_models():
            try:
                self._age_net    = cv2.dnn.readNet(_AGE_MODEL,    _AGE_PROTO)
                self._gender_net = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
                print("[FaceAnalyzer] ✓ Age/Gender DNN models ready")
            except Exception as e:
                print(f"[FaceAnalyzer] Age/Gender load failed: {e}")
        else:
            print("[FaceAnalyzer] Age/Gender models unavailable — showing 'Face detected' only")

    # ── Main capture loop ──────────────────────────────────────────────────────

    def _loop(self):
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            if self._frame_count % config.FACE_ANALYSIS_EVERY_N_FRAMES == 0:
                results = self._analyze(frame, haar)
                results = results if results else self._prev_results
                self._prev_results = results
            else:
                with self._lock:
                    results = list(self.latest_results)

            annotated = self._draw(frame, results)

            with self._lock:
                self.latest_frame   = annotated
                self.latest_results = results

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _analyze(self, frame, haar):
        """Detect faces then predict age/gender (if models loaded)."""
        boxes = self._detect_faces_ssd(frame) if self._face_net else \
                self._detect_faces_haar(frame, haar)

        if not boxes:
            return []

        results = []
        h, w = frame.shape[:2]

        for (x1, y1, bw, bh) in boxes:
            # Pad the crop for better age/gender accuracy
            pad  = 20
            cx1  = max(0, x1 - pad)
            cy1  = max(0, y1 - pad)
            cx2  = min(w, x1 + bw + pad)
            cy2  = min(h, y1 + bh + pad)
            crop = frame[cy1:cy2, cx1:cx2]

            if self._age_net and self._gender_net and crop.size > 0:
                age, gender, conf = self._predict_age_gender(crop)
            else:
                age, gender, conf = "?", "Unknown", 100.0

            results.append({
                "age":        age,
                "gender":     gender,
                "confidence": conf,
                "box":        (x1, y1, bw, bh),
            })

        return results

    # ── Face detection: SSD ───────────────────────────────────────────────────

    def _detect_faces_ssd(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._face_net.setInput(blob)
        dets = self._face_net.forward()

        boxes = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.55:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            bw, bh = x2 - x1, y2 - y1
            if bw >= 40 and bh >= 40:
                boxes.append((x1, y1, bw, bh))
        return boxes

    # ── Face detection: Haar (fallback) ───────────────────────────────────────

    def _detect_faces_haar(self, frame, cascade):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces] if len(faces) > 0 else []

    # ── Age/Gender prediction ─────────────────────────────────────────────────

    def _predict_age_gender(self, face_crop):
        blob = cv2.dnn.blobFromImage(
            face_crop, 1.0, (227, 227), _MODEL_MEAN, swapRB=False
        )
        # Gender
        self._gender_net.setInput(blob)
        gp       = self._gender_net.forward()[0]
        gender   = GENDER_LABELS[int(np.argmax(gp))]
        gconf    = round(float(np.max(gp)) * 100, 1)

        # Age
        self._age_net.setInput(blob)
        ap       = self._age_net.forward()[0]
        age      = AGE_BUCKETS[int(np.argmax(ap))]

        return age, gender, gconf

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame, results):
        out = frame.copy()
        for r in results:
            x, y, w, h = r["box"]
            gender = r["gender"]
            age    = r["age"]
            conf   = r["confidence"]

            is_male = gender.lower() == "male"
            # Green for male, magenta for female, cyan for unknown
            color = (0, 210, 0) if is_male else \
                    (210, 0, 210) if gender != "Unknown" else (0, 200, 200)

            _draw_corner_box(out, x, y, w, h, color)

            if gender == "Unknown":
                label = "Face detected"
            else:
                label = f"{gender}  {age}  {conf:.0f}%"
            _draw_pill_label(out, label, x, y, color)

        return out


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_corner_box(img, x, y, w, h, color, t=2, cl=18):
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, 1)
    for cx, cy, dx, dy in [(x, y, 1, 1), (x2, y, -1, 1),
                            (x, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * cl, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * cl), color, t)


def _draw_pill_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    p  = 5
    y0 = max(y - th - p * 2, 0)
    cv2.rectangle(img, (x, y0), (x + tw + p * 2, y0 + th + p * 2), color, -1)
    cv2.putText(img, text, (x + p, y0 + th + p - 1),
                font, 0.55, (255, 255, 255), 1)
