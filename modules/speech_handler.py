"""
modules/speech_handler.py
─────────────────────────
Handles:
  • Microphone audio capture (push-to-record style)
  • Saving raw audio to a temp WAV file
  • Transcribing with OpenAI Whisper (offline, CPU-friendly)
"""

import os
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import whisper

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class SpeechHandler:
    """
    Push-to-record speech recognizer.

    Usage:
        handler = SpeechHandler(source_lang="te")
        handler.load_model()       # call once at startup (slow)

        handler.start_recording()  # user presses Record button
        handler.stop_recording()   # user releases / presses Stop → returns transcript text
    """

    def __init__(self, source_lang: str = config.DEFAULT_SOURCE_LANG):
        self.source_lang  = source_lang
        self._model       = None
        self._recording   = False
        self._audio_chunks = []
        self._stream      = None
        self._lock        = threading.Lock()

        # Ensure output dirs exist
        os.makedirs(os.path.dirname(config.TEMP_AUDIO_FILE), exist_ok=True)

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, on_progress=None):
        """
        Load the Whisper model (downloads on first run, cached afterwards).
        Call this in a background thread to avoid freezing the UI.
        on_progress: optional callable(str) for status messages.
        """
        if on_progress:
            on_progress(f"Loading Whisper '{config.WHISPER_MODEL_SIZE}' model…")
        self._model = whisper.load_model(config.WHISPER_MODEL_SIZE)
        if on_progress:
            on_progress("Whisper model ready ✓")

    # ── Recording ──────────────────────────────────────────────────────────────

    def start_recording(self):
        """Begin capturing audio from the microphone."""
        if self._recording:
            return
        self._audio_chunks = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_recording(self) -> str:
        """
        Stop capturing audio, save WAV, transcribe, return transcript text.
        Blocks until transcription is complete.
        """
        if not self._recording:
            return ""

        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        return self._transcribe()

    def is_recording(self) -> bool:
        return self._recording

    # ── Internal ───────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        with self._lock:
            self._audio_chunks.append(indata.copy())

    def _transcribe(self) -> str:
        """Concatenate captured chunks, write WAV, run Whisper."""
        if not self._audio_chunks:
            return ""

        with self._lock:
            audio_data = np.concatenate(self._audio_chunks, axis=0).flatten()

        # Save to temp WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_write(config.TEMP_AUDIO_FILE, config.AUDIO_SAMPLE_RATE, audio_int16)

        # Transcribe with Whisper
        if self._model is None:
            return "[Whisper model not loaded yet]"

        result = self._model.transcribe(
            config.TEMP_AUDIO_FILE,
            language=self.source_lang,   # e.g. "te" for Telugu
            fp16=False,                  # CPU-safe (fp16 needs CUDA)
            verbose=False,
        )
        return result.get("text", "").strip()
