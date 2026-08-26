"""
face_analyzer_patch.py
──────────────────────
After training on Kaggle, download age_estimator_indian.onnx
and place it in:  models/age_estimator_indian.onnx

Then in face_analyzer.py:
  1. Add _load_custom_age_model() call in _load_insightface()
  2. Replace age extraction with _predict_age_custom()

This gives Indian-optimised age prediction instead of
InsightFace's default genderage model.
"""

# ── PASTE THIS into FaceAnalyzer class ──────────────────────────────────────

# In __init__(), add:
#   self._custom_age_sess = None   # ONNX age model

# In _load_insightface(), after self._if_app = app, add:
#   self._load_custom_age_model()

# ── New method to add ────────────────────────────────────────────────────────

CUSTOM_AGE_MODEL = os.path.join(_MODELS_DIR, "age_estimator_indian.onnx")

def _load_custom_age_model(self):
    """Load custom Indian-optimised age estimator (ONNX)."""
    if not os.path.exists(CUSTOM_AGE_MODEL):
        print("[FaceAnalyzer] Custom age model not found — using InsightFace default")
        return
    try:
        import onnxruntime as ort
        self._custom_age_sess = ort.InferenceSession(
            CUSTOM_AGE_MODEL,
            providers=["CPUExecutionProvider"]
        )
        print("[FaceAnalyzer] Custom Indian age model loaded OK")
    except Exception as e:
        print(f"[FaceAnalyzer] Custom age model load failed: {e}")

def _predict_age_custom(self, face_crop_bgr):
    """
    Run custom ONNX age model on a face crop.
    face_crop_bgr: numpy (H, W, 3) BGR uint8
    Returns: int age in years
    """
    import cv2
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Preprocess — same as training
    img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,224,224)

    result = self._custom_age_sess.run(["age"], {"face_crop": img})
    age = float(result[0][0])
    return max(1, min(90, int(round(age))))   # clamp to realistic range


# ── In _analyze_insightface(), replace the age extraction with: ──────────────
#
# OLD:
#   age = int(getattr(best, "age", 0))
#
# NEW:
#   if self._custom_age_sess is not None:
#       # Crop face region and run custom Indian model
#       h_frame, w_frame = age_frame.shape[:2]
#       pad = 10
#       crop = age_frame[
#           max(0, y1-pad):min(h_frame, y2+pad),
#           max(0, x1-pad):min(w_frame, x2+pad)
#       ]
#       age = self._predict_age_custom(crop) if crop.size > 0 else int(getattr(best, "age", 0))
#   else:
#       age = int(getattr(best, "age", 0))   # fallback to InsightFace
