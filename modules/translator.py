"""
modules/translator.py
─────────────────────
Handles:
  • Text translation between any two supported languages
    using deep-translator (Google Translate backend).
  • Requires internet connection.
"""

from deep_translator import GoogleTranslator


class Translator:
    """
    Wraps GoogleTranslator for easy source→target translation.

    Example:
        t = Translator(source="te", target="hi")
        hindi_text = t.translate("నమస్కారం")
    """

    def __init__(self,
                 source: str = "te",
                 target: str = "hi"):
        self.source = source
        self.target = target
        self._translator = self._build()

    # ── Public API ──────────────────────────────────────────────────────────────

    def update_languages(self, source: str, target: str):
        """Update source/target language and rebuild translator."""
        self.source = source
        self.target = target
        self._translator = self._build()

    def translate(self, text: str) -> str:
        """
        Translate text from source language to target language.
        Returns translated string, or original on error.
        """
        if not text or not text.strip():
            return ""
        try:
            return self._translator.translate(text.strip())
        except Exception as e:
            return f"[Translation error: {e}]"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build(self):
        return GoogleTranslator(source=self.source, target=self.target)
