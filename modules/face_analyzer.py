"""
modules/face_analyzer.py  v3
────────────────────────────
Completely rewritten to fix silent DeepFace/mediapipe failures on Python 3.13.

Strategy:
  1. OpenCV Haar Cascade for face DETECTION  (always works, built-in)
  2. OpenCV DNN (Caffe models) for AGE/GENDER estimation
     - Uses the well-known Levi & Hassner age/gender CNN models
     - Auto-downloads model files on first run to models/ directory
  3. Smooth results across frames (no flickering)
"""

import cv2
import os
import threading
import urllib.request
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ── Model URLs (Levi & Hassner pre-trained Caffe models) ─────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

_FACE_PROTO   = os.path.join(_MODELS_DIR, "deploy.prototxt")
_FACE_MODEL   = os.path.join(_MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
_AGE_PROTO    = os.path.join(_MODELS_DIR, "age_deploy.prototxt")
_AGE_MODEL    = os.path.join(_MODELS_DIR, "age_net.caffemodel")
_GENDER_PROTO = os.path.join(_MODELS_DIR, "gender_deploy.prototxt")
_GENDER_MODEL = os.path.join(_MODELS_DIR, "gender_net.caffemodel")

_MODEL_URLS = {
    _FACE_PROTO:   "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    _FACE_MODEL:   "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    _AGE_PROTO:    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/age_deploy.prototxt",
    _AGE_MODEL:    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_gender_models/age_net.caffemodel",
    _GENDER_PROTO: "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/gender_deploy.prototxt",
    _GENDER_MODEL: "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_gender_models/gender_net.caffemodel",
}

AGE_BUCKETS    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                  '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS  = ['Male', 'Female']

# Mean values for model preprocessing (ImageNet mean)
_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


def _ensure_models():
    """Download model files if not present. Called once at startup."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    for path, url in _MODEL_URLS.items():
        if not os.path.exists(path):
            fname = os.path.basename(path)
            print(f"[FaceAnalyzer] Downloading {fname}…")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[FaceAnalyzer] ✓ {fname} downloaded")
            except Exception as e:
                print(f"[FaceAnalyzer] ✗ Failed to download {fname}: {e}")
                return False
    return True


class FaceAnalyzer:
    """
    Threaded face analyzer using OpenCV DNN (no DeepFace, no mediapipe).
    Detects faces with SSD ResNet, estimates age/gender with Caffe CNNs.
    """

    def __init__(self):
        self._cap        = None
        self._thread     = None
        self._running    = False
        self._lock       = threading.Lock()
        self._frame_count = 0
        self._models_ok  = False

        # DNN models (loaded on start)
        self._face_net   = None
        self._age_net    = None
        self._gender_net = None

        # Smooth results across frames
        self.latest_frame   = None
        self.latest_results = []
        self._prev_results  = []

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self):
        """Download models if needed, open webcam, start analysis thread."""
        print("[FaceAnalyzer] Checking model files…")
        self._models_ok = _ensure_models()

        if self._models_ok:
            try:
                self._face_net   = cv2.dnn.readNet(_FACE_MODEL,   _FACE_PROTO)
                self._age_net    = cv2.dnn.readNet(_AGE_MODEL,    _AGE_PROTO)
                self._gender_net = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
                print("[FaceAnalyzer] ✓ DNN models loaded")
            except Exception as e:
                print(f"[FaceAnalyzer] DNN load failed: {e} — using Haar fallback")
                self._models_ok = False

        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Check WEBCAM_INDEX in config.py")

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

    # ── Internal Loop ───────────────────────────────────────────────────────────

    def _loop(self):
        # Haar cascade fallback (always available in OpenCV)
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            if self._frame_count % config.FACE_ANALYSIS_EVERY_N_FRAMES == 0:
                if self._models_ok and self._face_net:
                    results = self._analyze_dnn(frame)
                else:
                    results = self._analyze_haar(frame, haar)

                results = results if results else self._prev_results
                self._prev_results = results
            else:
                with self._lock:
                    results = list(self.latest_results)

            annotated = self._draw(frame, results)

            with self._lock:
                self.latest_frame   = annotated
                self.latest_results = results

    # ── DNN Analysis (SSD + Age/Gender CNNs) ───────────────────────────────────

    def _analyze_dnn(self, frame):
        """Detect faces + predict age/gender using OpenCV DNN models."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.6:   # face detection confidence threshold
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            face_w, face_h = x2 - x1, y2 - y1
            if face_w < 30 or face_h < 30:
                continue

            # Padding around face for better age/gender accuracy
            pad = 20
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            face_crop = frame[y1p:y2p, x1p:x2p]

            age, gender, gender_conf = self._predict_age_gender(face_crop)

            results.append({
                "age":        age,
                "gender":     gender,
                "confidence": round(gender_conf * 100, 1),
                "box":        (x1, y1, face_w, face_h),
            })
        return results

    def _predict_age_gender(self, face_img):
        """Run age and gender CNNs on a face crop."""
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), _MODEL_MEAN, swapRB=False
        )

        # Gender
        self._gender_net.setInput(blob)
        gender_preds = self._gender_net.forward()[0]
        gender_idx   = int(np.argmax(gender_preds))
        gender       = GENDER_LABELS[gender_idx]
        gender_conf  = float(gender_preds[gender_idx])

        # Age
        self._age_net.setInput(blob)
        age_preds = self._age_net.forward()[0]
        age_label = AGE_BUCKETS[int(np.argmax(age_preds))]

        return age_label, gender, gender_conf

    # ── Haar Cascade Fallback ──────────────────────────────────────────────────

    def _analyze_haar(self, frame, cascade):
        """Simple Haar cascade detection — no age/gender, just bounding box."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(60, 60))
        results = []
        for (x, y, fw, fh) in faces:
            results.append({
                "age":        "?",
                "gender":     "Unknown",
                "confidence": 100.0,
                "box":        (x, y, fw, fh),
            })
        return results

    # ── Drawing ─────────────────────────────────────────────────────────────────

    def _draw(self, frame, results):
        out = frame.copy()
        for r in results:
            x, y, w, h = r["box"]
            gender = r["gender"]
            age    = r["age"]
            conf   = r.get("confidence", 0)

            is_male = gender.lower() == "male"
            color   = (0, 210, 0) if is_male else (210, 0, 210)

            # Corner accent box
            _draw_corner_box(out, x, y, w, h, color)

            # Label
            label = f"{'Male' if is_male else gender}  {age}  {conf:.0f}%"
            _draw_pill_label(out, label, x, y, color)

        return out


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_corner_box(img, x, y, w, h, color, t=2, cl=20):
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, 1)
    for cx, cy, dx, dy in [(x, y, 1, 1), (x2, y, -1, 1),
                            (x, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * cl, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * cl), color, t)


def _draw_pill_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    p = 5
    y0 = max(y - th - p * 2, 0)
    cv2.rectangle(img, (x, y0), (x + tw + p * 2, y0 + th + p * 2), color, -1)
    cv2.putText(img, text, (x + p, y0 + th + p - 1), font, 0.55, (255, 255, 255), 1)
