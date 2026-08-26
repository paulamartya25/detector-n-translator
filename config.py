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

# ── Whisper Settings (battery-friendly defaults) ───────────────────────────────
# Sizes: tiny < base < small < medium < large
# Recommendation for degraded battery / CPU-only:
#   "tiny"  → very fast, less accurate
#   "base"  → good balance for most Indian languages
WHISPER_MODEL_SIZE = "base"

# ── Face Analysis Settings ─────────────────────────────────────────────────────
# Analyze every Nth frame to reduce CPU load (higher = less frequent = less power)
FACE_ANALYSIS_EVERY_N_FRAMES = 5

# Webcam index (0 = default built-in camera)
WEBCAM_INDEX = 0

# ── Output Settings ────────────────────────────────────────────────────────────
# "transcript" → show translated text in UI panel
# "audio"      → save translated speech as .mp3
# "both"       → do both (toggle in UI)
DEFAULT_OUTPUT_MODE = "both"

TRANSCRIPT_OUTPUT_DIR = "outputs/transcripts"
AUDIO_OUTPUT_DIR      = "outputs/audio"

# ── Audio Recording Settings ───────────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 16000   # Hz — required by Whisper
AUDIO_CHANNELS     = 1       # Mono
TEMP_AUDIO_FILE    = "outputs/_temp_recording.wav"
