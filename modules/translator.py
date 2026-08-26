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
        Automatically chunks long texts to stay within Google Translate limits.
        Returns translated string, or original on error.
        """
        if not text or not text.strip():
            return ""
        text = text.strip()

        # Google Translate free API limit is ~4500 chars per request
        MAX_CHARS = 4500
        if len(text) <= MAX_CHARS:
            return self._translate_chunk(text)

        # Split long text into sentence-based chunks
        import re
        sentences = re.split(r'(?<=[।.!?])\s+', text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= MAX_CHARS:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        translated_parts = [self._translate_chunk(c) for c in chunks]
        return " ".join(translated_parts)

    def _translate_chunk(self, text: str) -> str:
        """Translate a single chunk (under the API char limit)."""
        try:
            return self._translator.translate(text)
        except Exception as e:
            return f"[Translation error: {e}]"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build(self):
        return GoogleTranslator(source=self.source, target=self.target)
