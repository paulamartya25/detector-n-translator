"""
modules/translator.py
─────────────────────
v3 — Multi-backend translation with automatic fallback chain:
  1. GoogleTranslator (deep-translator)   — primary
  2. MyMemoryTranslator (deep-translator) — free, 1000 words/day, no API key
  3. Returns original text               — last resort (no crash)

Chunking for long texts is retained.
"""

import re
from deep_translator import GoogleTranslator, MyMemoryTranslator


class Translator:
    """
    Robust translator with a 3-level fallback chain.

    Example:
        t = Translator(source="te", target="hi")
        result = t.translate("నమస్కారం")
    """

    # Google Translate free API char limit per request
    _MAX_CHARS = 4500

    def __init__(self, source: str = "te", target: str = "hi"):
        self.source = source
        self.target = target
        self._build()

    # ── Public API ──────────────────────────────────────────────────────────────

    def update_languages(self, source: str, target: str):
        self.source = source
        self.target = target
        self._build()

    def translate(self, text: str) -> str:
        """
        Translate with auto-chunking for long texts.
        Returns translated string (never raises).
        """
        if not text or not text.strip():
            return ""
        text = text.strip()

        if len(text) <= self._MAX_CHARS:
            return self._translate_with_fallback(text)

        # Chunk by sentence boundary
        sentences = re.split(r'(?<=[।.!?])\s+', text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self._MAX_CHARS:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        return " ".join(self._translate_with_fallback(c) for c in chunks)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _build(self):
        """Build both translator backends."""
        try:
            self._google = GoogleTranslator(
                source=self.source, target=self.target
            )
        except Exception:
            self._google = None

        try:
            # MyMemory uses "en-GB" style codes; map ISO 639-1 → lang pair
            src = self.source
            tgt = self.target
            self._mymemory = MyMemoryTranslator(source=src, target=tgt)
        except Exception:
            self._mymemory = None

    def _translate_with_fallback(self, text: str) -> str:
        """
        Try translators in order:
          1. GoogleTranslator
          2. MyMemoryTranslator
          3. Return original text (never fail)
        """
        # ── 1. Google Translate ────────────────────────────────────────────────
        if self._google:
            try:
                result = self._google.translate(text)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                print(f"[Translator] Google failed: {e}")

        # ── 2. MyMemory Translate ──────────────────────────────────────────────
        if self._mymemory:
            try:
                result = self._mymemory.translate(text)
                if result and result.strip():
                    return f"{result.strip()}  [via MyMemory]"
            except Exception as e:
                print(f"[Translator] MyMemory failed: {e}")

        # ── 3. Last resort — return original ──────────────────────────────────
        print("[Translator] All backends failed — returning original text")
        return f"[No translation] {text}"
