---
title: Detector N Translator
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 3.50.2
app_file: app_gradio.py
pinned: false
license: mit
---

# Detector N Translator

Real-time multimodal AI: **Face Analysis** (age & gender detection) combined with **Multilingual Speech Translation** across 11 Indian languages.

## Features
- Live webcam face detection with bounding boxes
- Custom EfficientNetB3 model trained on UTKFace (Indian faces optimised)
- OpenAI Whisper speech recognition
- Translation with 3-level fallback (Google -> MyMemory -> Pons)
- gTTS audio output in target language

## How to Use
1. **Face Analysis tab** - Allow camera access and point your webcam at your face
2. **Speech Translation tab** - Select languages, record your voice, click Translate
