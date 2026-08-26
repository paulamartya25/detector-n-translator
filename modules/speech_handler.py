"""
modules/speech_handler.py
─────────────────────────
Improvements in v2:
  • Voice Activity Detection (VAD) — auto-detects when speech starts and ends
    using RMS energy threshold — no button press needed
  • Push-to-record mode still supported as alternative
  • Smooth energy-based silence detection with configurable timeout
  • on_vad_transcript callback: called with transcript when speech segment ends
"""

import os
import time
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
    Dual-mode speech recognizer:

    Mode 1 — Push-to-record (manual):
        handler.start_recording()
        handler.stop_recording()  → returns transcript str

    Mode 2 — Auto VAD (automatic):
        handler.start_vad(on_transcript=callback)
        handler.stop_vad()
        # callback(transcript_str) is called after each speech segment

    Always call load_model() once at startup before recording.
    """

    def __init__(self, source_lang: str = config.DEFAULT_SOURCE_LANG):
        self.source_lang   = source_lang
        self._model        = None
        self._lock         = threading.Lock()

        # ── Push-to-record state ──────────────────────────────────────────────
        self._recording      = False
        self._audio_chunks   = []
        self._stream         = None

        # ── VAD state ────────────────────────────────────────────────────────
        self._vad_active     = False
        self._vad_thread     = None
        self._vad_callback   = None   # callable(str)

        # Ensure temp output dir exists
        os.makedirs(os.path.dirname(config.TEMP_AUDIO_FILE), exist_ok=True)

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, on_progress=None):
        """
        Load Whisper model (downloads once, cached afterwards).
        Call in a background thread — blocks until done.
        """
        if on_progress:
            on_progress(f"⏳  Loading Whisper '{config.WHISPER_MODEL_SIZE}' model…")
        self._model = whisper.load_model(config.WHISPER_MODEL_SIZE)
        if on_progress:
            on_progress("✅  Whisper model ready")

    # ══════════════════════════════════════════════════════════════════════════
    #  Mode 1 — Push-to-Record
    # ══════════════════════════════════════════════════════════════════════════

    def start_recording(self):
        """Begin capturing microphone audio (push-to-record mode)."""
        if self._recording or self._vad_active:
            return
        self._audio_chunks = []
        self._recording    = True
        self._stream = sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            callback=self._push_callback,
        )
        self._stream.start()

    def stop_recording(self) -> str:
        """
        Stop capture, transcribe, return text.
        Blocks until Whisper finishes.
        """
        if not self._recording:
            return ""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._transcribe(self._audio_chunks)

    def is_recording(self) -> bool:
        return self._recording

    def _push_callback(self, indata, frames, time_info, status):
        with self._lock:
            self._audio_chunks.append(indata.copy())

    # ══════════════════════════════════════════════════════════════════════════
    #  Mode 2 — Voice Activity Detection (Auto)
    # ══════════════════════════════════════════════════════════════════════════

    def start_vad(self, on_transcript):
        """
        Start listening continuously.
        When speech is detected, record it.
        When silence exceeds VAD_SILENCE_DURATION seconds, transcribe
        and call on_transcript(text).

        on_transcript: callable(str) — called in a background thread.
        """
        if self._vad_active or self._recording:
            return
        self._vad_active   = True
        self._vad_callback = on_transcript
        self._vad_thread   = threading.Thread(
            target=self._vad_loop, daemon=True
        )
        self._vad_thread.start()

    def stop_vad(self):
        """Stop the VAD listener."""
        self._vad_active = False
        if self._vad_thread:
            self._vad_thread.join(timeout=5)
        self._vad_thread = None

    def is_vad_active(self) -> bool:
        return self._vad_active

    def _vad_loop(self):
        """
        Core VAD logic:
        - Continuously read 100ms audio blocks
        - Track RMS energy
        - When energy > threshold → "speech detected" → accumulate
        - When energy drops below threshold for > SILENCE_DURATION → stop & transcribe
        """
        CHUNK_DURATION   = 0.1   # seconds per block
        CHUNK_SAMPLES    = int(config.AUDIO_SAMPLE_RATE * CHUNK_DURATION)
        ENERGY_THRESHOLD = config.VAD_ENERGY_THRESHOLD
        SILENCE_TIMEOUT  = config.VAD_SILENCE_DURATION  # seconds of silence to stop

        speech_chunks    = []
        silence_counter  = 0.0
        in_speech        = False

        with sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:

            while self._vad_active:
                block, overflowed = stream.read(CHUNK_SAMPLES)
                block = block.flatten()
                rms   = float(np.sqrt(np.mean(block ** 2)))

                if rms > ENERGY_THRESHOLD:
                    # ── Speech activity ───────────────────────────────────────
                    if not in_speech:
                        in_speech = True
                    silence_counter = 0.0
                    speech_chunks.append(block)

                else:
                    # ── Silence ───────────────────────────────────────────────
                    if in_speech:
                        speech_chunks.append(block)  # include tail silence
                        silence_counter += CHUNK_DURATION

                        if silence_counter >= SILENCE_TIMEOUT:
                            # End of speech segment — transcribe
                            chunks_to_transcribe = list(speech_chunks)
                            speech_chunks   = []
                            silence_counter = 0.0
                            in_speech       = False

                            # Transcribe in a separate thread so VAD keeps listening
                            threading.Thread(
                                target=self._vad_transcribe_and_callback,
                                args=(chunks_to_transcribe,),
                                daemon=True,
                            ).start()

    def _vad_transcribe_and_callback(self, chunks):
        """Transcribe captured chunks and fire the callback."""
        text = self._transcribe(chunks)
        if text and self._vad_callback:
            self._vad_callback(text)

    # ── Shared Transcription ───────────────────────────────────────────────────

    def _transcribe(self, chunks: list) -> str:
        """
        Concatenate audio chunks, save WAV, run Whisper, return clean text.

        Filters applied (in order):
          1. Skip clips shorter than VAD_MIN_SPEECH_DURATION seconds
          2. Discard if Whisper's no_speech_prob exceeds threshold
          3. Discard if text looks like hallucinated garbage
        """
        if not chunks:
            return ""

        audio_data = np.concatenate(chunks, axis=0).flatten()

        # ── Guard 1: minimum duration ─────────────────────────────────────────
        duration = len(audio_data) / config.AUDIO_SAMPLE_RATE
        if duration < config.VAD_MIN_SPEECH_DURATION:
            print(f"[Speech] Clip too short ({duration:.1f}s) — skipped")
            return ""

        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_write(config.TEMP_AUDIO_FILE, config.AUDIO_SAMPLE_RATE, audio_int16)

        if self._model is None:
            return "[Model not loaded]"

        result = self._model.transcribe(
            config.TEMP_AUDIO_FILE,
            language=self.source_lang,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,  # reduces hallucination chaining
        )

        # ── Guard 2: no_speech_prob check ────────────────────────────────────
        # Whisper scores each segment; average them
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            if avg_no_speech > config.WHISPER_NO_SPEECH_THRESHOLD:
                print(f"[Speech] Discarded — no_speech_prob={avg_no_speech:.2f} "
                      f"(threshold={config.WHISPER_NO_SPEECH_THRESHOLD})")
                return ""

        text = result.get("text", "").strip()

        # ── Guard 3: hallucination filter ─────────────────────────────────────
        if _is_hallucination(text):
            print(f"[Speech] Discarded hallucination: '{text[:60]}'")
            return ""

        return text


def _is_hallucination(text: str) -> bool:
    """
    Detect common Whisper hallucination patterns:
    - Text is too short (single word from noise)
    - Contains high ratio of non-target-script characters
    - Matches known Whisper noise phrases
    """
    if not text:
        return True

    # Known Whisper hallucination phrases (appear when fed silence/noise)
    KNOWN_HALLUCINATIONS = {
        "you", "thank you", "thanks", "thank you.", "thanks.",
        "www.", ".com", "subtitles", "subscribe", "like and subscribe",
        "bye", "bye.", "okay", "ok.", "hmm", "um", "uh",
    }
    if text.lower().strip(".!? ") in KNOWN_HALLUCINATIONS:
        return True

    # Too short — less than 3 words is likely noise
    words = text.split()
    if len(words) < 2:
        return True

    # High proportion of non-Latin / mixed-script garbage characters
    # (real speech in one language shouldn't mix 4+ scripts)
    import unicodedata
    scripts = set()
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch, "")
                if "LATIN" in name:
                    scripts.add("LATIN")
                elif "ARABIC" in name:
                    scripts.add("ARABIC")
                elif "HANGUL" in name:
                    scripts.add("HANGUL")
                elif "CYRILLIC" in name:
                    scripts.add("CYRILLIC")
                elif "CJK" in name:
                    scripts.add("CJK")
                elif "THAI" in name:
                    scripts.add("THAI")
                elif "DEVANAGARI" in name:
                    scripts.add("DEVANAGARI")
                elif "TELUGU" in name:
                    scripts.add("TELUGU")
            except Exception:
                pass
    # If 3+ different scripts appear in one utterance → garbage
    if len(scripts) >= 3:
        return True

    return False

