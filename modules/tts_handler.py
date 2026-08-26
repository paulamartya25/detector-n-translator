"""
modules/tts_handler.py
──────────────────────
Handles:
  • Converting translated text to speech using gTTS
  • Saving audio as .mp3 in the outputs/audio directory
  • Playing the audio via pygame
  • Saving transcripts as .txt files
"""

import os
import datetime
import threading
import pygame
from gtts import gTTS

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class TTSHandler:
    """
    Text-to-speech and transcript saver.

    Usage:
        tts = TTSHandler()
        tts.speak_and_save("नमस्ते, आप कैसे हैं?", lang="hi")
        tts.save_transcript("original", "translated", src_lang="te", tgt_lang="hi")
    """

    def __init__(self):
        os.makedirs(config.AUDIO_OUTPUT_DIR,      exist_ok=True)
        os.makedirs(config.TRANSCRIPT_OUTPUT_DIR,  exist_ok=True)
        pygame.mixer.init()
        self._playing = False

    # ── Audio (TTS) ─────────────────────────────────────────────────────────────

    def speak_and_save(self,
                       text: str,
                       lang: str = config.DEFAULT_TARGET_LANG,
                       play: bool = True) -> str:
        """
        Convert text to speech, save as .mp3, optionally play it.
        Returns the saved file path.
        Runs in a background thread so UI stays responsive.
        """
        filepath = self._generate_filepath(lang)
        thread = threading.Thread(
            target=self._tts_worker,
            args=(text, lang, filepath, play),
            daemon=True,
        )
        thread.start()
        return filepath

    def stop_playback(self):
        """Stop currently playing audio."""
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        self._playing = False

    # ── Transcript ──────────────────────────────────────────────────────────────

    def save_transcript(self,
                        original: str,
                        translated: str,
                        src_lang: str,
                        tgt_lang: str) -> str:
        """
        Append a transcript entry to a dated .txt file.
        Returns the saved file path.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        date_str  = datetime.datetime.now().strftime("%Y-%m-%d")
        filepath  = os.path.join(config.TRANSCRIPT_OUTPUT_DIR, f"transcript_{date_str}.txt")

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

    def _tts_worker(self, text: str, lang: str, filepath: str, play: bool):
        """Background worker: generate TTS, save, optionally play."""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(filepath)

            if play:
                self._playing = True
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                # Wait until done
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(5)
                self._playing = False
        except Exception as e:
            print(f"[TTSHandler] Error: {e}")

    def _generate_filepath(self, lang: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"translation_{lang}_{timestamp}.mp3"
        return os.path.join(config.AUDIO_OUTPUT_DIR, filename)
