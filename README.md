# Face Analysis + Speech Translator  v2

A real-time desktop app that simultaneously does three things:
1. **Detects faces** — shows age (exact years) and gender via InsightFace
2. **Transcribes speech** — using OpenAI Whisper `small` model
3. **Translates speech** — between 11 Indian + world languages with audio output

---

## Features

| Feature | Technology | Details |
|---|---|---|
| Face Detection | InsightFace SCRFD | State-of-the-art, handles backlit faces |
| Age Estimation | InsightFace Genderage | Exact age in years (e.g. `~27y`) |
| Gender | InsightFace Genderage | Male / Female + confidence % |
| Speech Recognition | OpenAI Whisper `small` | 85% accuracy, works offline |
| Auto Speech Detection | VAD (energy-based) | No button press needed |
| Translation | Google Translate (3 fallbacks) | Works even when API rate-limited |
| Text-to-Speech | pyttsx3 (offline) + gTTS | Instant playback + .mp3 saved |
| Audio Output | pygame | Save to file + play on demand |

---

## Supported Languages

Telugu, Hindi, English, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu

---

## Quick Start

### 1. Setup (run once)
```
setup.bat
```
Or manually:
```
pip install -r requirements.txt
```

### 2. Run
```
python app.py
```
Or double-click `run.bat`

---

## First Run Downloads (automatic, one-time only)

| Model | Size | Purpose |
|---|---|---|
| Whisper `small` | ~460 MB | Speech recognition |
| InsightFace `buffalo_l` | ~200 MB | Face detection + age/gender |
| OpenCV SSD ResNet | ~10 MB | Face detection fallback |

All models are cached permanently — subsequent runs start instantly.

---

## How to Use

### Auto Listen (VAD mode) — Recommended
1. Set **Source** (speaker's language) and **Target** (your language)
2. Click **AUTO LISTEN (VAD)**
3. Speak naturally — app auto-detects speech start/end
4. Transcript + translation appear automatically
5. Green bar appears → click **▶ Play** to hear audio

### Manual Record mode
1. Click **HOLD TO RECORD (Manual)** → speak → click again to stop
2. Results appear in the Transcript panel

### VAD Tuning
- **Sensitivity slider** — raise if background noise triggers false detections
- **Silence timeout** — seconds of silence before finalising a speech segment
- Watch the **mic energy bar** — your voice should clearly spike above the threshold

---

## Output Files

All saved to the `outputs/` folder:

| Folder | Contents |
|---|---|
| `outputs/transcripts/` | Daily `.txt` files with timestamped transcripts |
| `outputs/audio/` | `.mp3` files of translated speech |

Click **📂 Open outputs/** in the app to open the folder directly.

---

## Project Structure

```
├── app.py                  Main application (Tkinter UI)
├── main.py                 Alias entry point
├── config.py               All user-adjustable settings
├── modules/
│   ├── face_analyzer.py    InsightFace + OpenCV DNN face detection
│   ├── speech_handler.py   Whisper transcription + VAD
│   ├── translator.py       Google Translate with 3-level fallback
│   └── tts_handler.py      pyttsx3 + gTTS text-to-speech
├── models/                 OpenCV DNN model files (auto-downloaded)
├── outputs/
│   ├── audio/              Saved .mp3 translations
│   └── transcripts/        Saved .txt transcripts
├── setup.bat               One-click Windows setup
└── run.bat                 One-click Windows launcher
```

---

## Configuration (`config.py`)

```python
WHISPER_MODEL_SIZE        = "small"    # tiny/base/small/medium
FACE_ANALYSIS_EVERY_N_FRAMES = 3       # lower = more frequent
FACE_CONFIDENCE_THRESHOLD = 35         # min % to show prediction
VAD_ENERGY_THRESHOLD      = 0.025      # raise if getting noise, lower if missing speech
VAD_SILENCE_DURATION      = 2.0        # seconds of silence to end segment
VAD_MIN_SPEECH_DURATION   = 1.0        # min clip length to transcribe
WHISPER_NO_SPEECH_THRESHOLD = 0.5      # discard if Whisper thinks silence
```

---

## Fallback Chains

### Face Detection
```
InsightFace SCRFD (best)
    → OpenCV SSD ResNet (if InsightFace unavailable)
        → Haar Cascade (always works, zero download)
```

### Translation
```
Google Translate (deep-translator)
    → MyMemoryTranslator (free, no API key)
        → Direct HTTP to translate.googleapis.com
            → Return original text (never crashes)
```

### Text-to-Speech
```
pyttsx3 offline (instant, no internet)
    → gTTS online (saves .mp3, plays via pygame)
```

---

## Requirements

- Python 3.9+  (tested on 3.13)
- Windows 10/11
- Webcam
- Microphone
- Internet (for translation + first-run model downloads)

---

## GitHub

**[paulamartya25/detector-n-translator](https://github.com/paulamartya25/detector-n-translator)**
