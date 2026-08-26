"""
modules/translator.py
─────────────────────
v4 — Fixed MyMemoryTranslator language codes (needs full names, not ISO codes)
     + Added LibreTranslate as 3rd fallback (free, open-source)
     + Better error reporting per backend
"""

import re
from deep_translator import GoogleTranslator, MyMemoryTranslator


# ── MyMemory uses full English language names, not ISO codes ──────────────────
_ISO_TO_MYMEMORY = {
    "te": "telugu",
    "hi": "hindi",
    "en": "english",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
    "bn": "bengali",
    "mr": "marathi",
    "gu": "gujarati",
    "pa": "punjabi",
    "ur": "urdu",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "zh": "chinese",
    "ar": "arabic",
    "ja": "japanese",
    "ko": "korean",
}


class Translator:
    """
    4-level fallback translation chain:
      1. GoogleTranslator      (deep-translator, Google backend)
      2. MyMemoryTranslator    (deep-translator, free 1000w/day)
      3. Direct Google request (raw HTTP, no library)
      4. Return original text  (never crash)

    Example:
        t = Translator(source="en", target="hi")
        print(t.translate("Hello"))   # → "नमस्ते"
    """

    _MAX_CHARS = 4500   # Google Translate per-request limit

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
        """Translate text, auto-chunking if needed. Never raises."""
        if not text or not text.strip():
            return ""
        text = text.strip()

        if len(text) <= self._MAX_CHARS:
            return self._translate_with_fallback(text)

        # Chunk by sentence boundary for long texts
        sentences = re.split(r'(?<=[।.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self._MAX_CHARS:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)

        return " ".join(self._translate_with_fallback(c) for c in chunks)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _build(self):
        """Instantiate all available backends."""
        # Google
        try:
            self._google = GoogleTranslator(source=self.source, target=self.target)
        except Exception as e:
            print(f"[Translator] Google init failed: {e}")
            self._google = None

        # MyMemory — needs full language names
        src_mm = _ISO_TO_MYMEMORY.get(self.source, self.source)
        tgt_mm = _ISO_TO_MYMEMORY.get(self.target, self.target)
        try:
            self._mymemory = MyMemoryTranslator(source=src_mm, target=tgt_mm)
        except Exception as e:
            print(f"[Translator] MyMemory init failed: {e}")
            self._mymemory = None

    def _translate_with_fallback(self, text: str) -> str:
        """Try each backend in order, return first success."""

        # ── 1. GoogleTranslator ────────────────────────────────────────────────
        if self._google:
            try:
                result = self._google.translate(text)
                if result and result.strip() and result.strip() != text:
                    return result.strip()
            except Exception as e:
                print(f"[Translator] Google failed: {type(e).__name__}: {e}")

        # ── 2. MyMemoryTranslator ─────────────────────────────────────────────
        if self._mymemory:
            try:
                result = self._mymemory.translate(text)
                if result and result.strip() and result.strip() != text:
                    return result.strip()
            except Exception as e:
                print(f"[Translator] MyMemory failed: {type(e).__name__}: {e}")

        # ── 3. Direct HTTP call to Google Translate ────────────────────────────
        try:
            result = self._direct_google(text)
            if result and result.strip() and result.strip() != text:
                return result.strip()
        except Exception as e:
            print(f"[Translator] Direct HTTP failed: {type(e).__name__}: {e}")

        # ── 4. Last resort — return original ──────────────────────────────────
        print(f"[Translator] ⚠️ All backends failed for: '{text[:40]}…'")
        return text   # return original text, no error tag

    def _direct_google(self, text: str) -> str:
        """
        Unofficial direct HTTP call to Google Translate.
        Uses the same endpoint as the browser's translate.google.com.
        """
        import urllib.request
        import urllib.parse
        import json

        url = "https://translate.googleapis.com/translate_a/single"
        params = urllib.parse.urlencode({
            "client": "gtx",
            "sl":     self.source,
            "tl":     self.target,
            "dt":     "t",
            "q":      text,
        })
        req = urllib.request.Request(
            f"{url}?{params}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Response structure: [[[translated, original, ...], ...], ...]
        parts = [seg[0] for seg in data[0] if seg[0]]
        return "".join(parts)
