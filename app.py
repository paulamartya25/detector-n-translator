"""
app.py
──────
Main entry point for the Face Analysis + Speech Translation System.

Layout:
  ┌─────────────────────────┬────────────────────────────────┐
  │   Live Webcam Feed      │   Speech & Translation Panel   │
  │  (age / gender overlay) │                                │
  │                         │  [Source Lang]  [Target Lang]  │
  │                         │  [Output Mode: Transcript/Both]│
  │                         │                                │
  │                         │  ● START RECORDING             │
  │                         │  ■ STOP  RECORDING             │
  │                         │                                │
  │                         │  ── Transcript ──              │
  │                         │  [scrollable text area]        │
  │                         │                                │
  │                         │  Status: Ready                 │
  └─────────────────────────┴────────────────────────────────┘
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2

# Local modules
from modules.face_analyzer  import FaceAnalyzer
from modules.speech_handler import SpeechHandler
from modules.translator     import Translator
from modules.tts_handler    import TTSHandler
import config


# ── Color palette ────────────────────────────────────────────────────────────
BG_COLOR       = "#1e1e2e"   # dark background
PANEL_COLOR    = "#2a2a3e"   # slightly lighter panels
ACCENT         = "#7c3aed"   # purple accent
ACCENT_HOVER   = "#6d28d9"
REC_COLOR      = "#dc2626"   # red for recording
STOP_COLOR     = "#16a34a"   # green for stop
TEXT_COLOR     = "#e2e8f0"   # light text
SUBTEXT_COLOR  = "#94a3b8"   # muted text
FONT_MAIN      = ("Segoe UI", 10)
FONT_TITLE     = ("Segoe UI", 12, "bold")
FONT_MONO      = ("Consolas", 10)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Analysis + Speech Translator")
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Module instances ─────────────────────────────────────────────────
        self._face_analyzer  = FaceAnalyzer()
        self._speech_handler = None    # built after language selection
        self._translator     = None    # built after language selection
        self._tts_handler    = TTSHandler()

        # ── State ────────────────────────────────────────────────────────────
        self._webcam_running  = False
        self._model_loaded    = False
        self._recording       = False

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

        # ── Start background tasks ────────────────────────────────────────────
        self._start_webcam()
        self._load_whisper_async()

    # ════════════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ════════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=ACCENT, pady=6)
        title_bar.pack(fill=tk.X)
        tk.Label(
            title_bar,
            text="🎭  Face Analysis + Speech Translator",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg="white",
        ).pack(side=tk.LEFT, padx=12)

        # ── Main content frame ────────────────────────────────────────────────
        content = tk.Frame(self, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left column: webcam
        self._build_webcam_panel(content)

        # Right column: controls + transcript
        self._build_control_panel(content)

    # ── Webcam Panel ──────────────────────────────────────────────────────────

    def _build_webcam_panel(self, parent):
        left = tk.Frame(parent, bg=BG_COLOR)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            left, text="📷  Live Feed",
            font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR
        ).pack(anchor=tk.W, pady=(0, 4))

        cam_frame = tk.Frame(left, bg="#000000", bd=2, relief=tk.SUNKEN)
        cam_frame.pack(fill=tk.BOTH, expand=True)

        self._cam_label = tk.Label(cam_frame, bg="#000000")
        self._cam_label.pack(fill=tk.BOTH, expand=True)

        # Age/gender info bar below video
        self._face_info_var = tk.StringVar(value="No face detected")
        tk.Label(
            left,
            textvariable=self._face_info_var,
            font=FONT_MONO, bg=PANEL_COLOR, fg=ACCENT,
            anchor=tk.W, padx=8, pady=4,
        ).pack(fill=tk.X, pady=(4, 0))

    # ── Control / Translation Panel ───────────────────────────────────────────

    def _build_control_panel(self, parent):
        right = tk.Frame(parent, bg=BG_COLOR, width=360)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))
        right.pack_propagate(False)

        # ── Language selection ────────────────────────────────────────────────
        lang_frame = tk.LabelFrame(
            right, text=" 🌐  Language Settings ",
            font=FONT_MAIN, bg=PANEL_COLOR, fg=TEXT_COLOR,
            bd=1, labelanchor=tk.NW, padx=8, pady=8,
        )
        lang_frame.pack(fill=tk.X, pady=(0, 8))

        lang_names = list(config.SUPPORTED_LANGUAGES.keys())

        # Source language
        tk.Label(lang_frame, text="Source (Speaker):", font=FONT_MAIN,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=0, column=0, sticky=tk.W, pady=2)
        self._src_lang_var = tk.StringVar(value="Telugu")
        src_combo = ttk.Combobox(
            lang_frame, textvariable=self._src_lang_var,
            values=lang_names, state="readonly", width=14,
        )
        src_combo.grid(row=0, column=1, padx=(8, 0), pady=2, sticky=tk.W)

        # Target language
        tk.Label(lang_frame, text="Target (You):", font=FONT_MAIN,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=1, column=0, sticky=tk.W, pady=2)
        self._tgt_lang_var = tk.StringVar(value="Hindi")
        tgt_combo = ttk.Combobox(
            lang_frame, textvariable=self._tgt_lang_var,
            values=lang_names, state="readonly", width=14,
        )
        tgt_combo.grid(row=1, column=1, padx=(8, 0), pady=2, sticky=tk.W)

        # Apply button
        tk.Button(
            lang_frame, text="Apply",
            font=FONT_MAIN, bg=ACCENT, fg="white",
            activebackground=ACCENT_HOVER, relief=tk.FLAT,
            command=self._apply_language_settings, cursor="hand2",
        ).grid(row=2, column=1, sticky=tk.E, pady=(6, 0))

        # ── Output mode ───────────────────────────────────────────────────────
        mode_frame = tk.LabelFrame(
            right, text=" 🔊  Output Mode ",
            font=FONT_MAIN, bg=PANEL_COLOR, fg=TEXT_COLOR,
            bd=1, labelanchor=tk.NW, padx=8, pady=8,
        )
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        self._output_mode_var = tk.StringVar(value="both")
        modes = [("Transcript only", "transcript"),
                 ("Audio only",      "audio"),
                 ("Both",            "both")]
        for i, (label, value) in enumerate(modes):
            tk.Radiobutton(
                mode_frame, text=label, variable=self._output_mode_var,
                value=value, font=FONT_MAIN,
                bg=PANEL_COLOR, fg=TEXT_COLOR,
                selectcolor=PANEL_COLOR, activebackground=PANEL_COLOR,
            ).grid(row=0, column=i, padx=4)

        # ── Recording controls ────────────────────────────────────────────────
        rec_frame = tk.Frame(right, bg=BG_COLOR)
        rec_frame.pack(fill=tk.X, pady=(0, 8))

        self._rec_btn = tk.Button(
            rec_frame,
            text="🎙  START RECORDING",
            font=("Segoe UI", 11, "bold"),
            bg=REC_COLOR, fg="white",
            activebackground="#b91c1c",
            relief=tk.FLAT, padx=10, pady=8,
            command=self._toggle_recording,
            cursor="hand2",
        )
        self._rec_btn.pack(fill=tk.X)

        self._rec_status_var = tk.StringVar(value="⬤  Idle")
        tk.Label(
            rec_frame,
            textvariable=self._rec_status_var,
            font=FONT_MAIN, bg=BG_COLOR, fg=SUBTEXT_COLOR,
        ).pack(anchor=tk.W, pady=(2, 0))

        # ── Transcript box ────────────────────────────────────────────────────
        tk.Label(
            right, text="📝  Transcript / Translation",
            font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(anchor=tk.W, pady=(4, 2))

        self._transcript_box = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD, font=FONT_MONO,
            bg=PANEL_COLOR, fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT, padx=8, pady=6,
            state=tk.DISABLED,
        )
        self._transcript_box.pack(fill=tk.BOTH, expand=True)

        tk.Button(
            right, text="Clear",
            font=FONT_MAIN, bg=PANEL_COLOR, fg=SUBTEXT_COLOR,
            relief=tk.FLAT, command=self._clear_transcript,
            cursor="hand2",
        ).pack(anchor=tk.E, pady=(4, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="⏳  Loading Whisper model…")
        tk.Label(
            right,
            textvariable=self._status_var,
            font=FONT_MAIN, bg=BG_COLOR, fg=SUBTEXT_COLOR,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(6, 0))

    # ════════════════════════════════════════════════════════════════════════════
    #  Webcam Loop
    # ════════════════════════════════════════════════════════════════════════════

    def _start_webcam(self):
        self._face_analyzer.start()
        self._webcam_running = True
        self._update_webcam()

    def _update_webcam(self):
        """Called every ~33ms via after() to refresh the webcam frame."""
        if not self._webcam_running:
            return

        frame, results = self._face_analyzer.get_latest()

        if frame is not None:
            # Convert BGR → RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((480, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            self._cam_label.configure(image=photo)
            self._cam_label.image = photo  # keep reference

            # Update face info bar
            if results:
                parts = [f"Person {i+1}: {r['gender']}, ~{r['age']} yrs"
                         for i, r in enumerate(results)]
                self._face_info_var.set("  |  ".join(parts))
            else:
                self._face_info_var.set("No face detected")

        self.after(33, self._update_webcam)   # ~30 FPS

    # ════════════════════════════════════════════════════════════════════════════
    #  Whisper Model Loading
    # ════════════════════════════════════════════════════════════════════════════

    def _load_whisper_async(self):
        """Load Whisper model in a background thread."""
        src_lang = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(),
                                                   config.DEFAULT_SOURCE_LANG)
        self._speech_handler = SpeechHandler(source_lang=src_lang)
        self._translator     = Translator(
            source=src_lang,
            target=config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(),
                                                   config.DEFAULT_TARGET_LANG),
        )

        def _load():
            self._speech_handler.load_model(
                on_progress=lambda msg: self.after(0, self._set_status, msg)
            )
            self._model_loaded = True
            self.after(0, self._set_status, "✅  Ready — model loaded")

        threading.Thread(target=_load, daemon=True).start()

    # ════════════════════════════════════════════════════════════════════════════
    #  Recording & Translation
    # ════════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self._model_loaded:
            messagebox.showwarning("Not Ready", "Whisper model is still loading. Please wait.")
            return
        self._recording = True
        self._rec_btn.configure(text="⏹  STOP RECORDING", bg=STOP_COLOR,
                                activebackground="#15803d")
        self._rec_status_var.set("🔴  Recording…")
        self._set_status("Recording… speak now")
        self._speech_handler.start_recording()

    def _stop_recording(self):
        self._recording = False
        self._rec_btn.configure(text="🎙  START RECORDING", bg=REC_COLOR,
                                activebackground="#b91c1c")
        self._rec_status_var.set("⏳  Processing…")
        self._set_status("Transcribing…")

        # Run stop + transcribe + translate in background thread
        threading.Thread(target=self._process_recording, daemon=True).start()

    def _process_recording(self):
        """Background: stop → transcribe → translate → output."""
        # 1. Transcribe
        original = self._speech_handler.stop_recording()
        if not original:
            self.after(0, self._set_status, "⚠️  No speech detected")
            self.after(0, self._rec_status_var.set, "⬤  Idle")
            return

        self.after(0, self._set_status, "Translating…")

        # 2. Translate
        translated = self._translator.translate(original) if self._translator else original

        # 3. Output
        src_code = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), "te")
        tgt_code = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), "hi")
        mode     = self._output_mode_var.get()

        # Always append to transcript UI
        self.after(0, self._append_transcript, original, translated, src_code, tgt_code)

        if mode in ("transcript", "both"):
            saved = self._tts_handler.save_transcript(original, translated, src_code, tgt_code)
            self.after(0, self._set_status, f"✅  Transcript saved → {os.path.basename(saved)}")

        if mode in ("audio", "both"):
            audio_path = self._tts_handler.speak_and_save(translated, lang=tgt_code, play=True)
            self.after(0, self._set_status,
                       f"✅  Audio saved → {os.path.basename(audio_path)}")

        self.after(0, self._rec_status_var.set, "⬤  Idle")

    # ════════════════════════════════════════════════════════════════════════════
    #  Language Settings
    # ════════════════════════════════════════════════════════════════════════════

    def _apply_language_settings(self):
        src_name = self._src_lang_var.get()
        tgt_name = self._tgt_lang_var.get()
        src_code = config.SUPPORTED_LANGUAGES.get(src_name, "en")
        tgt_code = config.SUPPORTED_LANGUAGES.get(tgt_name, "en")

        if src_code == tgt_code:
            messagebox.showwarning("Same Language",
                                   "Source and target language must be different.")
            return

        if self._speech_handler:
            self._speech_handler.source_lang = src_code
        if self._translator:
            self._translator.update_languages(src_code, tgt_code)

        self._set_status(f"✅  Languages updated: {src_name} → {tgt_name}")

    # ════════════════════════════════════════════════════════════════════════════
    #  Transcript UI Helpers
    # ════════════════════════════════════════════════════════════════════════════

    def _append_transcript(self, original, translated, src_code, tgt_code):
        self._transcript_box.configure(state=tk.NORMAL)
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._transcript_box.insert(tk.END,
            f"\n[{ts}]\n"
            f"🎤 ({src_code}): {original}\n"
            f"🌐 ({tgt_code}): {translated}\n"
            f"{'─' * 40}\n"
        )
        self._transcript_box.configure(state=tk.DISABLED)
        self._transcript_box.see(tk.END)

    def _clear_transcript(self):
        self._transcript_box.configure(state=tk.NORMAL)
        self._transcript_box.delete("1.0", tk.END)
        self._transcript_box.configure(state=tk.DISABLED)

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    # ════════════════════════════════════════════════════════════════════════════
    #  Cleanup
    # ════════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._webcam_running = False
        self._face_analyzer.stop()
        if self._speech_handler and self._speech_handler.is_recording():
            self._speech_handler.stop_recording()
        self._tts_handler.stop_playback()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
