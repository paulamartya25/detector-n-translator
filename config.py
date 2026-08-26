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
# Lower this value if face is not being detected (try 40 first, then 30)
FACE_CONFIDENCE_THRESHOLD = 40   # 40% confidence required

# ── VAD (Voice Activity Detection) Settings ────────────────────────────────────
# RMS energy threshold: values above this = speech, below = silence
# Typical quiet room: 0.010–0.020   Normal room: 0.025–0.05
# RAISE this if you see garbage/rubbish transcriptions from background noise
VAD_ENERGY_THRESHOLD = 0.025   # raised from 0.010 — filters ambient room noise

# Seconds of continuous silence after which a speech segment is finalized
VAD_SILENCE_DURATION = 1.5

# Minimum audio duration (seconds) to even attempt transcription
# Clips shorter than this are almost always noise bursts — skip them
VAD_MIN_SPEECH_DURATION = 1.5

# Whisper no_speech_prob threshold (0.0–1.0)
# If Whisper says there's > this probability of NO speech → discard result
# 0.5 = discard if Whisper is more than 50% sure it heard silence/noise
WHISPER_NO_SPEECH_THRESHOLD = 0.5

# ── Output Settings ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_MODE     = "both"
TRANSCRIPT_OUTPUT_DIR   = "outputs/transcripts"
AUDIO_OUTPUT_DIR        = "outputs/audio"

# ── Audio Recording Settings ───────────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 16000   # Hz — required by Whisper
AUDIO_CHANNELS     = 1       # Mono
TEMP_AUDIO_FILE    = "outputs/_temp_recording.wav"
