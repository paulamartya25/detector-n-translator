"""
modules/face_analyzer.py  v5  — InsightFace + ONNX Runtime
------------------------------------------------------------
Best accuracy pipeline:
  Face detection  -> SCRFD-10GF        (InsightFace, state-of-the-art 2023)
  Age/Gender      -> Genderage model   (InsightFace buffalo_l pack)

Fallback chain if InsightFace unavailable:
  -> OpenCV SSD ResNet (already downloaded)
  -> OpenCV Haar Cascade (zero-download, built in)

InsightFace auto-downloads its models (~200 MB) to ~/.insightface/models/
on first run.
"""

import cv2
import os
import threading
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# -- Try InsightFace -----------------------------------------------------------
_INSIGHTFACE_OK = False
try:
    import insightface
    from insightface.app import FaceAnalysis as _InsightFaceApp
    _INSIGHTFACE_OK = True
    print("[FaceAnalyzer] InsightFace available OK")
except ImportError:
    print("[FaceAnalyzer] InsightFace not available — using OpenCV DNN fallback")

# -- OpenCV DNN fallback paths -------------------------------------------------
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_FACE_PROTO = os.path.join(_MODELS_DIR, "deploy.prototxt")
_FACE_MODEL = os.path.join(_MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Age/Gender labels (for OpenCV DNN fallback)
AGE_BUCKETS   = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                 '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']
_MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)

# -- Age/Gender DNN fallback model URLs ---------------------------------------
_AG_URLS = {
    os.path.join(_MODELS_DIR, "age_deploy.prototxt"): [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    ],
    os.path.join(_MODELS_DIR, "age_net.caffemodel"): [
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    ],
    os.path.join(_MODELS_DIR, "gender_deploy.prototxt"): [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    ],
    os.path.join(_MODELS_DIR, "gender_net.caffemodel"): [
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
    ],
}


class FaceAnalyzer:
    """
    Threaded face analyzer.
    Uses InsightFace (best accuracy) if available, else OpenCV DNN.
    """

    def __init__(self):
        self._cap         = None
        self._thread      = None
        self._running     = False
        self._lock        = threading.Lock()
        self._frame_count = 0

        # InsightFace
        self._if_app      = None

        # OpenCV DNN fallback
        self._face_net    = None
        self._age_net     = None
        self._gender_net  = None

        self.latest_frame   = None
        self.latest_results = []
        self._prev_results  = []

    # -- Public API -------------------------------------------------------------

    def start(self):
        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check WEBCAM_INDEX in config.py")

        if _INSIGHTFACE_OK:
            # Load InsightFace in background (downloads models on first run)
            threading.Thread(target=self._load_insightface, daemon=True).start()
        else:
            # Load OpenCV DNN models
            threading.Thread(target=self._load_opencv_dnn, daemon=True).start()

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

    # -- InsightFace Loader ----------------------------------------------------

    def _load_insightface(self):
        """Load InsightFace buffalo_l model pack (downloads ~200MB on first run)."""
        try:
            print("[FaceAnalyzer] Loading InsightFace models (first run downloads ~200MB)…")
            app = _InsightFaceApp(
                name="buffalo_l",          # best accuracy pack
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0,          # 0 = CPU
                        det_size=(640, 640),
                        det_thresh=0.45)
            self._if_app = app
            print("[FaceAnalyzer] OK InsightFace ready — SCRFD + Age/Gender loaded")
        except Exception as e:
            print(f"[FaceAnalyzer] InsightFace load failed: {e}")
            print("[FaceAnalyzer] Falling back to OpenCV DNN…")
            self._load_opencv_dnn()

    # -- OpenCV DNN Loader (fallback) ------------------------------------------

    def _load_opencv_dnn(self):
        """Load OpenCV SSD + Age/Gender Caffe models."""
        import urllib.request

        # Face detector (already downloaded)
        if os.path.exists(_FACE_MODEL) and os.path.exists(_FACE_PROTO):
            try:
                self._face_net = cv2.dnn.readNet(_FACE_MODEL, _FACE_PROTO)
                print("[FaceAnalyzer] OK SSD face detector ready")
            except Exception as e:
                print(f"[FaceAnalyzer] SSD load failed: {e}")

        # Age/Gender — download if missing
        all_present = all(os.path.exists(p) for p in _AG_URLS)
        if not all_present:
            for path, urls in _AG_URLS.items():
                if not os.path.exists(path):
                    for url in urls:
                        try:
                            fname = os.path.basename(path)
                            print(f"[FaceAnalyzer] Downloading {fname}…")
                            urllib.request.urlretrieve(url, path)
                            size = os.path.getsize(path) / 1e6
                            print(f"[FaceAnalyzer] OK {fname} ({size:.1f} MB)")
                            break
                        except Exception as e:
                            print(f"[FaceAnalyzer] Download failed: {e}")

        age_p    = os.path.join(_MODELS_DIR, "age_deploy.prototxt")
        age_m    = os.path.join(_MODELS_DIR, "age_net.caffemodel")
        gender_p = os.path.join(_MODELS_DIR, "gender_deploy.prototxt")
        gender_m = os.path.join(_MODELS_DIR, "gender_net.caffemodel")

        if all(os.path.exists(p) for p in [age_p, age_m, gender_p, gender_m]):
            try:
                self._age_net    = cv2.dnn.readNet(age_m, age_p)
                self._gender_net = cv2.dnn.readNet(gender_m, gender_p)
                print("[FaceAnalyzer] OK Age/Gender Caffe models ready")
            except Exception as e:
                print(f"[FaceAnalyzer] Age/Gender load failed: {e}")

    # -- Main capture loop ------------------------------------------------------

    @property
    def status(self):
        """Return a human-readable status string for the UI."""
        if self._if_app:
            return "InsightFace ready"
        if self._face_net:
            return "OpenCV DNN ready"
        return "Loading face models..."

    def _enhance(self, frame):
        """
        CLAHE contrast enhancement — fixes backlit / dark face conditions.
        Converts to LAB, enhances L channel, converts back to BGR.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
                # Apply CLAHE enhancement to help with backlit / dark faces
                enhanced = self._enhance(frame)
                try:
                    if self._if_app:
                        results = self._analyze_insightface(enhanced)
                    elif self._face_net:
                        results = self._analyze_opencv_dnn(enhanced, haar)
                    else:
                        results = self._analyze_haar(enhanced, haar)
                except Exception as e:
                    print(f"[FaceAnalyzer] Analysis error: {e}")
                    results = []

                results = results if results else self._prev_results
                self._prev_results = results
            else:
                with self._lock:
                    results = list(self.latest_results)

            # Draw on original frame (not enhanced) so colours look natural
            annotated = self._draw(frame, results)
            with self._lock:
                self.latest_frame   = annotated
                self.latest_results = results

    # -- InsightFace Analysis --------------------------------------------------

    def _analyze_insightface(self, frame):
        """Run InsightFace SCRFD detector + Genderage model."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self._if_app.get(rgb)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:
                continue

            age        = int(getattr(face, "age",    0))
            gender_raw = getattr(face, "gender", 0)
            det_score  = float(getattr(face, "det_score", 1.0))

            if isinstance(gender_raw, str):
                gender = "Male" if gender_raw.upper() in ("M", "MALE") else "Female"
            else:
                gender = "Male" if int(gender_raw) == 1 else "Female"

            results.append({
                "age":        f"~{age}y",
                "gender":     gender,
                "confidence": round(det_score * 100, 1),
                "box":        (x1, y1, w, h),
            })
        return results

    # -- OpenCV DNN Analysis (fallback) ----------------------------------------

    def _analyze_opencv_dnn(self, frame, haar):
        """SSD ResNet face detection + Caffe age/gender."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123)
        )
        self._face_net.setInput(blob)
        dets = self._face_net.forward()

        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.45:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            bw, bh = x2 - x1, y2 - y1
            if bw < 35 or bh < 35:
                continue

            pad  = 20
            crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

            if self._age_net and self._gender_net and crop.size > 0:
                age, gender, gconf = self._predict_ag(crop)
            else:
                age, gender, gconf = "?", "Unknown", round(conf * 100, 1)

            results.append({
                "age": age, "gender": gender,
                "confidence": gconf, "box": (x1, y1, bw, bh),
            })
        return results

    def _predict_ag(self, crop):
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), _MODEL_MEAN, swapRB=False)
        self._gender_net.setInput(blob)
        gp     = self._gender_net.forward()[0]
        gender = GENDER_LABELS[int(np.argmax(gp))]
        gconf  = round(float(np.max(gp)) * 100, 1)
        self._age_net.setInput(blob)
        ap  = self._age_net.forward()[0]
        age = AGE_BUCKETS[int(np.argmax(ap))]
        return age, gender, gconf

    # -- Haar Cascade (last resort) --------------------------------------------

    def _analyze_haar(self, frame, cascade):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        return [{"age": "?", "gender": "Unknown",
                 "confidence": 100.0, "box": (x, y, w, h)}
                for (x, y, w, h) in faces] if len(faces) > 0 else []

    # -- Drawing ---------------------------------------------------------------

    def _draw(self, frame, results):
        out = frame.copy()
        for r in results:
            x, y, w, h = r["box"]
            gender = r["gender"]
            age    = r["age"]
            conf   = r["confidence"]

            is_male = gender.lower() == "male"
            color   = (0, 210, 0) if is_male else \
                      (210, 0, 210) if gender != "Unknown" else (0, 200, 200)

            _draw_corner_box(out, x, y, w, h, color)
            label = f"{gender}  {age}  {conf:.0f}%" \
                    if gender != "Unknown" else "Face detected"
            _draw_pill_label(out, label, x, y, color)
        return out


# -- Drawing helpers -----------------------------------------------------------

def _draw_corner_box(img, x, y, w, h, color, t=2, cl=18):
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, 1)
    for cx, cy, dx, dy in [(x,y,1,1),(x2,y,-1,1),(x,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx, cy), (cx + dx*cl, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy*cl), color, t)


def _draw_pill_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    p  = 5
    y0 = max(y - th - p*2, 0)
    cv2.rectangle(img, (x, y0), (x + tw + p*2, y0 + th + p*2), color, -1)
    cv2.putText(img, text, (x+p, y0+th+p-1), font, 0.55, (255,255,255), 1)
