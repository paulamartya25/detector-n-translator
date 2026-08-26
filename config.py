# config.py — User-adjustable settings for the app

# ── Language Settings ──────────────────────────────────────────────────────────
DEFAULT_SOURCE_LANG = "te"   # Telugu (ISO 639-1 code)
DEFAULT_TARGET_LANG = "hi"   # Hindi  (ISO 639-1 code)

# Supported languages shown in the dropdowns (display name → ISO code)
SUPPORTED_LANGUAGES = {
    "Telugu":    "te",
    "Hindi":     "hi",
    "English":   "en",
    "Tamil":     "ta",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Bengali":   "bn",
    "Marathi":   "mr",
    "Gujarati":  "gu",
    "Punjabi":   "pa",
    "Urdu":      "ur",
}

# ── Whisper Settings ───────────────────────────────────────────────────────────
# Sizes: tiny < base < small < medium < large
# Recommendation for CPU-only / degraded battery: "base"
WHISPER_MODEL_SIZE = "base"

# ── Face Analysis Settings ─────────────────────────────────────────────────────
# Analyze every Nth frame (higher = less frequent = less battery drain)
FACE_ANALYSIS_EVERY_N_FRAMES = 5

# Webcam index (0 = default built-in camera)
WEBCAM_INDEX = 0

# Minimum gender confidence % to show a result (0–100)
# Below this threshold the detection is silently discarded
FACE_CONFIDENCE_THRESHOLD = 60   # 60% confidence required

# ── VAD (Voice Activity Detection) Settings ────────────────────────────────────
# RMS energy threshold: values above this = speech, below = silence
# Typical quiet room: 0.005–0.01   Loud room: 0.02–0.05
VAD_ENERGY_THRESHOLD = 0.01

# Seconds of continuous silence after which a speech segment is finalized
VAD_SILENCE_DURATION = 1.5

# ── Output Settings ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_MODE     = "both"
TRANSCRIPT_OUTPUT_DIR   = "outputs/transcripts"
AUDIO_OUTPUT_DIR        = "outputs/audio"

# ── Audio Recording Settings ───────────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 16000   # Hz — required by Whisper
AUDIO_CHANNELS     = 1       # Mono
TEMP_AUDIO_FILE    = "outputs/_temp_recording.wav"
