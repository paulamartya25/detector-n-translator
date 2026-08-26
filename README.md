# Face Analysis + Speech Translator 🎭🎙️

A real-time desktop app that does three things simultaneously:

1. **Age & Gender Detection** — detects faces from your webcam and estimates age and gender using DeepFace
2. **Speech Recognition** — records your speech (supports Telugu, Hindi, English, and many more) using OpenAI Whisper (works fully offline)
3. **Translation** — translates the recognized speech into your desired language (e.g., Telugu → Hindi) using Google Translate

---

## Features

- 🎥 Live webcam feed with face bounding boxes + age/gender labels
- 🎙️ Push-to-record microphone input
- 🌐 Dropdown language selection for source and target languages
- 📝 Live transcript display in the UI
- 🔊 Saves translated speech as `.mp3` audio files (using gTTS)
- 💾 Saves transcripts as `.txt` files with timestamps
- 🔋 Battery-friendly: face analysis every 5th frame, `base` Whisper model by default

---

## Supported Languages

Telugu, Hindi, English, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu

---

## Setup

### Requirements
- Python 3.9+
- Internet connection (for Google Translate and gTTS)
- Webcam + Microphone

### Install
Double-click `setup.bat` **or** run:
```bash
pip install -r requirements.txt
```

### Run
Double-click `run.bat` **or** run:
```bash
python app.py
```

---

## Project Structure

```
├── app.py                  # Main Tkinter UI
├── config.py               # All user settings
├── requirements.txt
├── setup.bat               # One-click setup
├── run.bat                 # One-click run
├── modules/
│   ├── face_analyzer.py    # DeepFace age/gender detection
│   ├── speech_handler.py   # Whisper speech recognition
│   ├── translator.py       # Google Translate
│   └── tts_handler.py      # gTTS audio + transcript saver
└── outputs/
    ├── audio/              # Saved .mp3 translation files
    └── transcripts/        # Saved .txt transcript files
```

---

## Configuration

Edit [`config.py`](config.py) to change:
- Default source/target language
- Whisper model size (`tiny` / `base` / `small` / `medium`)
- Face analysis frequency
- Webcam index

---

## Tech Stack

| Feature | Library |
|---|---|
| Face Detection & Age/Gender | `deepface` + `opencv-python` |
| Speech Recognition | `openai-whisper` |
| Translation | `deep-translator` |
| Text-to-Speech | `gTTS` |
| Audio I/O | `sounddevice`, `pygame` |
| UI | `tkinter` (built-in) |
