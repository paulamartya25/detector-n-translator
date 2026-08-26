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
#
#   tiny   → 39M params  | ~60% accuracy  | Very fast   (not recommended)
#   base   → 74M params  | ~75% accuracy  | Fast
#   small  → 244M params | ~85% accuracy  | ~2x base    ← NOW ACTIVE ✅
#   medium → 769M params | ~90% accuracy  | Too slow for live CPU use
#   large  → 1550M params| ~95% accuracy  | Needs GPU
#
# First run will auto-download the model (~460 MB for small) and cache it.
# Subsequent runs load from cache — no re-download needed.
WHISPER_MODEL_SIZE = "small"   # upgraded from "base" → better Telugu/Hindi accuracy

# ── Face Analysis Settings ─────────────────────────────────────────────────────
# Analyze every Nth frame (lower = more frequent = better but more CPU)
FACE_ANALYSIS_EVERY_N_FRAMES = 3   # was 5 → more responsive face updates

# Webcam index (0 = default built-in camera)
WEBCAM_INDEX = 0

# Minimum gender confidence % to show a result (0–100)
# Lower if face not detected, raise if getting wrong predictions
FACE_CONFIDENCE_THRESHOLD = 35   # was 40 → slightly more permissive

# ── VAD (Voice Activity Detection) Settings ────────────────────────────────────
# RMS energy threshold: values above this = speech, below = silence
# Typical quiet room: 0.015–0.020   Normal room: 0.025–0.035
# RAISE this if you still see garbage transcriptions from background noise
VAD_ENERGY_THRESHOLD = 0.025

# Seconds of continuous silence after which a speech segment is finalized
# Raise this if your speech is being cut off mid-sentence
VAD_SILENCE_DURATION = 2.0   # was 1.5 → more time for natural pauses

# Minimum audio duration (seconds) to even attempt transcription
# Clips shorter than this are almost always noise bursts — skip them
VAD_MIN_SPEECH_DURATION = 1.0   # was 1.5 → catches shorter phrases too

# Whisper no_speech_prob threshold (0.0–1.0)
# If Whisper says there's > this probability of NO speech → discard
WHISPER_NO_SPEECH_THRESHOLD = 0.5

# ── Output Settings ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_MODE     = "both"
TRANSCRIPT_OUTPUT_DIR   = "outputs/transcripts"
AUDIO_OUTPUT_DIR        = "outputs/audio"

# ── Audio Recording Settings ───────────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 16000   # Hz — required by Whisper
AUDIO_CHANNELS     = 1       # Mono
TEMP_AUDIO_FILE    = "outputs/_temp_recording.wav"
