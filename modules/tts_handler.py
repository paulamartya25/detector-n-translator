"""
modules/tts_handler.py
──────────────────────
Improvements in v2:
  • pyttsx3 (offline, instant, no internet) as PRIMARY TTS engine
  • gTTS (Google, online, better voice quality) as FALLBACK
  • pyttsx3 auto-selects best available voice for target language
  • Saves both .mp3 (gTTS) and speaks instantly (pyttsx3)
  • Transcript saving unchanged
"""

import os
import datetime
import threading
import pygame
from gtts import gTTS

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# pyttsx3 — offline TTS
try:
    import pyttsx3
    _PYTTSX3_OK = True
except ImportError:
    _PYTTSX3_OK = False
    print("[TTSHandler] pyttsx3 not found — falling back to gTTS only.")


# ── Language → pyttsx3 voice hint mapping ──────────────────────────────────────
# Maps ISO 639-1 codes to keywords to search in Windows SAPI voice names
_LANG_VOICE_HINTS = {
    "hi": ["hindi", "hemant", "kalpana"],
    "te": ["telugu"],
    "ta": ["tamil"],
    "kn": ["kannada"],
    "ml": ["malayalam"],
    "bn": ["bengali"],
    "mr": ["marathi"],
    "en": ["zira", "david", "english"],
    "gu": ["gujarati"],
    "pa": ["punjabi"],
    "ur": ["urdu"],
}


def _find_pyttsx3_voice(engine, lang_code: str):
    """Return a pyttsx3 voice ID matching the language hint, or None."""
    hints = _LANG_VOICE_HINTS.get(lang_code, [lang_code])
    voices = engine.getProperty("voices")
    for hint in hints:
        for v in voices:
            if hint.lower() in v.name.lower() or hint.lower() in v.id.lower():
                return v.id
    return None  # no matching voice found, will use system default


class TTSHandler:
    """
    Dual-engine TTS:
      1. pyttsx3 → offline, instant playback (primary)
      2. gTTS    → online, better quality, saves .mp3 (fallback / always saves)

    Usage:
        tts = TTSHandler()
        tts.speak_and_save("नमस्ते", lang="hi")
        tts.save_transcript("original", "translated", "te", "hi")
    """

    def __init__(self):
        os.makedirs(config.AUDIO_OUTPUT_DIR,     exist_ok=True)
        os.makedirs(config.TRANSCRIPT_OUTPUT_DIR, exist_ok=True)

        pygame.mixer.init()
        self._playing = False

        # Build pyttsx3 engine (one per process — not thread-safe, use lock)
        self._pyttsx3_engine = None
        self._pyttsx3_lock   = threading.Lock()
        if _PYTTSX3_OK:
            try:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 160)   # words per minute
                self._pyttsx3_engine.setProperty("volume", 1.0)
                print("[TTSHandler] pyttsx3 engine ready ✓")
            except Exception as e:
                print(f"[TTSHandler] pyttsx3 init failed: {e} — using gTTS only")
                self._pyttsx3_engine = None

    # ── Audio TTS ───────────────────────────────────────────────────────────────

    def speak_and_save(self,
                       text: str,
                       lang: str = config.DEFAULT_TARGET_LANG,
                       play: bool = False) -> str:
        """
        Convert text to speech and save as .mp3.
        play=False  → only saves the file (user can play manually)
        play=True   → saves AND plays immediately
        Returns the saved .mp3 filepath.
        """
        if not text or not text.strip():
            return ""

        filepath = self._generate_filepath(lang)
        threading.Thread(
            target=self._worker,
            args=(text, lang, filepath, play),
            daemon=True,
        ).start()
        return filepath

    def stop_playback(self):
        """Stop audio playback."""
        try:
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
        except Exception:
            pass
        self._playing = False

    # ── Transcript ──────────────────────────────────────────────────────────────

    def save_transcript(self,
                        original: str,
                        translated: str,
                        src_lang: str,
                        tgt_lang: str) -> str:
        """Append a timestamped entry to today's transcript file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        date_str  = datetime.datetime.now().strftime("%Y-%m-%d")
        filepath  = os.path.join(
            config.TRANSCRIPT_OUTPUT_DIR, f"transcript_{date_str}.txt"
        )
        entry = (
            f"\n[{timestamp}]\n"
            f"Original  ({src_lang}): {original}\n"
            f"Translated({tgt_lang}): {translated}\n"
            f"{'-' * 60}\n"
        )
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(entry)
        return filepath

    # ── Internal ────────────────────────────────────────────────────────────────

    def _worker(self, text: str, lang: str, filepath: str, play: bool):
        """
        Background worker:
          1. pyttsx3 for English/Latin-script only (instant, offline)
          2. gTTS for ALL languages — saves .mp3 and plays via pygame
             (gTTS supports Hindi, Telugu, Urdu, Tamil etc.)
        """
        # Languages that pyttsx3 can speak on Windows (has system voices)
        PYTTSX3_LANGS = {"en", "fr", "de", "es", "it", "pt"}
        spoken_by_pyttsx3 = False

        # ── Step 1: pyttsx3 for English/Latin only ────────────────────────────
        if play and self._pyttsx3_engine and lang in PYTTSX3_LANGS:
            with self._pyttsx3_lock:
                try:
                    voice_id = _find_pyttsx3_voice(self._pyttsx3_engine, lang)
                    if voice_id:
                        self._pyttsx3_engine.setProperty("voice", voice_id)
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
                    spoken_by_pyttsx3 = True
                    print(f"[TTSHandler] pyttsx3 spoke ({lang})")
                except Exception as e:
                    print(f"[TTSHandler] pyttsx3 failed: {e}")

        # ── Step 2: gTTS → .mp3 (all languages, always saved) ────────────────
        try:
            # Map ISO codes that gTTS needs differently
            GTTS_LANG_MAP = {
                "te": "te",  # Telugu
                "hi": "hi",  # Hindi
                "ur": "ur",  # Urdu
                "ta": "ta",  # Tamil
                "kn": "kn",  # Kannada
                "ml": "ml",  # Malayalam
                "bn": "bn",  # Bengali
                "mr": "mr",  # Marathi
                "gu": "gu",  # Gujarati
                "pa": "pa",  # Punjabi
            }
            gtts_lang = GTTS_LANG_MAP.get(lang, lang)
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(filepath)
            print(f"[TTSHandler] Saved audio: {os.path.basename(filepath)}")

            # Play via pygame if: play=True AND pyttsx3 didn't handle it
            if play and not spoken_by_pyttsx3:
                self._playing = True
                try:
                    pygame.mixer.music.load(filepath)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(5)
                    print(f"[TTSHandler] Playback done ({lang})")
                except Exception as pe:
                    print(f"[TTSHandler] pygame playback failed: {pe}")
                finally:
                    self._playing = False

        except Exception as e:
            print(f"[TTSHandler] gTTS error: {e}")

    def _generate_filepath(self, lang: str) -> str:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(config.AUDIO_OUTPUT_DIR, f"translation_{lang}_{ts}.mp3")
