"""
app.py  v2
──────────
Improvements:
  • VAD (Auto-listen) toggle — app listens continuously and auto-transcribes
  • Push-to-record still available for noisy environments
  • Mic energy level bar (live feedback while listening)
  • VAD status: Idle / Listening / Speech detected / Processing
  • Face confidence % shown in info bar
  • pyttsx3 offline TTS (instant playback)
"""

import os
import threading
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import sounddevice as sd

# Local modules
from modules.face_analyzer  import FaceAnalyzer
from modules.speech_handler import SpeechHandler
from modules.translator     import Translator
from modules.tts_handler    import TTSHandler
import config


# ── Color palette ──────────────────────────────────────────────────────────────
BG_COLOR       = "#1e1e2e"
PANEL_COLOR    = "#2a2a3e"
ACCENT         = "#7c3aed"
ACCENT_HOVER   = "#6d28d9"
REC_COLOR      = "#dc2626"
STOP_COLOR     = "#16a34a"
VAD_COLOR      = "#0ea5e9"     # blue for VAD/auto mode
TEXT_COLOR     = "#e2e8f0"
SUBTEXT_COLOR  = "#94a3b8"
SUCCESS_COLOR  = "#22c55e"
WARN_COLOR     = "#f59e0b"
FONT_MAIN      = ("Segoe UI", 10)
FONT_TITLE     = ("Segoe UI", 12, "bold")
FONT_MONO      = ("Consolas", 10)
FONT_SMALL     = ("Segoe UI", 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Analysis + Speech Translator  v2")
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Modules ──────────────────────────────────────────────────────────
        self._face_analyzer  = FaceAnalyzer()
        self._speech_handler = None
        self._translator     = None
        self._tts_handler    = TTSHandler()

        # ── State ────────────────────────────────────────────────────────────
        self._webcam_running  = False
        self._model_loaded    = False
        self._recording       = False     # push-to-record active
        self._vad_on          = False     # VAD auto mode active

        # ── Build UI + start services ─────────────────────────────────────────
        self._build_ui()
        self._start_webcam()
        self._load_whisper_async()
        self._start_energy_meter()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=ACCENT, pady=6)
        title_bar.pack(fill=tk.X)
        tk.Label(
            title_bar,
            text="🎭  Face Analysis + Speech Translator",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg="white",
        ).pack(side=tk.LEFT, padx=12)
        tk.Label(
            title_bar, text="v2",
            font=("Segoe UI", 9), bg=ACCENT, fg="#c4b5fd",
        ).pack(side=tk.LEFT)

        # Main content
        content = tk.Frame(self, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_webcam_panel(content)
        self._build_control_panel(content)

    # ── Webcam Panel ──────────────────────────────────────────────────────────

    def _build_webcam_panel(self, parent):
        left = tk.Frame(parent, bg=BG_COLOR)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            left, text="📷  Live Feed",
            font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(anchor=tk.W, pady=(0, 4))

        cam_frame = tk.Frame(left, bg="#000", bd=2, relief=tk.SUNKEN)
        cam_frame.pack(fill=tk.BOTH, expand=True)

        self._cam_label = tk.Label(cam_frame, bg="#000")
        self._cam_label.pack(fill=tk.BOTH, expand=True)

        # Face info bar
        self._face_info_var = tk.StringVar(value="No face detected")
        tk.Label(
            left,
            textvariable=self._face_info_var,
            font=FONT_MONO, bg=PANEL_COLOR, fg=ACCENT,
            anchor=tk.W, padx=8, pady=4,
        ).pack(fill=tk.X, pady=(4, 0))

        # Mic energy bar
        energy_row = tk.Frame(left, bg=BG_COLOR)
        energy_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            energy_row, text="🎤 Mic:",
            font=FONT_SMALL, bg=BG_COLOR, fg=SUBTEXT_COLOR,
        ).pack(side=tk.LEFT)
        self._energy_bar = ttk.Progressbar(
            energy_row, orient=tk.HORIZONTAL, length=200, mode="determinate"
        )
        self._energy_bar.pack(side=tk.LEFT, padx=6)
        self._energy_var = tk.StringVar(value="0.000")
        tk.Label(
            energy_row, textvariable=self._energy_var,
            font=FONT_SMALL, bg=BG_COLOR, fg=SUBTEXT_COLOR, width=5,
        ).pack(side=tk.LEFT)

    # ── Control Panel ─────────────────────────────────────────────────────────

    def _build_control_panel(self, parent):
        right = tk.Frame(parent, bg=BG_COLOR, width=370)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))
        right.pack_propagate(False)

        # ── Language settings ─────────────────────────────────────────────────
        lang_frame = tk.LabelFrame(
            right, text=" 🌐  Language Settings ",
            font=FONT_MAIN, bg=PANEL_COLOR, fg=TEXT_COLOR,
            bd=1, labelanchor=tk.NW, padx=8, pady=8,
        )
        lang_frame.pack(fill=tk.X, pady=(0, 6))

        lang_names = list(config.SUPPORTED_LANGUAGES.keys())

        tk.Label(lang_frame, text="Source (Speaker):", font=FONT_MAIN,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=0, column=0, sticky=tk.W, pady=2)
        self._src_lang_var = tk.StringVar(value="Telugu")
        ttk.Combobox(
            lang_frame, textvariable=self._src_lang_var,
            values=lang_names, state="readonly", width=14,
        ).grid(row=0, column=1, padx=(8, 0), pady=2, sticky=tk.W)

        tk.Label(lang_frame, text="Target (You):", font=FONT_MAIN,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=1, column=0, sticky=tk.W, pady=2)
        self._tgt_lang_var = tk.StringVar(value="Hindi")
        ttk.Combobox(
            lang_frame, textvariable=self._tgt_lang_var,
            values=lang_names, state="readonly", width=14,
        ).grid(row=1, column=1, padx=(8, 0), pady=2, sticky=tk.W)

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
            bd=1, labelanchor=tk.NW, padx=8, pady=6,
        )
        mode_frame.pack(fill=tk.X, pady=(0, 6))

        self._output_mode_var = tk.StringVar(value="both")
        for i, (lbl, val) in enumerate([
            ("Transcript", "transcript"),
            ("Audio only", "audio"),
            ("Both",       "both"),
        ]):
            tk.Radiobutton(
                mode_frame, text=lbl, variable=self._output_mode_var,
                value=val, font=FONT_MAIN,
                bg=PANEL_COLOR, fg=TEXT_COLOR,
                selectcolor=PANEL_COLOR, activebackground=PANEL_COLOR,
            ).grid(row=0, column=i, padx=4)

        # ── VAD settings ──────────────────────────────────────────────────────
        vad_settings = tk.LabelFrame(
            right, text=" 🎙  Voice Activity Detection ",
            font=FONT_MAIN, bg=PANEL_COLOR, fg=TEXT_COLOR,
            bd=1, labelanchor=tk.NW, padx=8, pady=6,
        )
        vad_settings.pack(fill=tk.X, pady=(0, 6))

        # Threshold slider
        tk.Label(vad_settings, text="Sensitivity:", font=FONT_SMALL,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=0, column=0, sticky=tk.W)
        self._vad_threshold_var = tk.DoubleVar(value=config.VAD_ENERGY_THRESHOLD)
        tk.Scale(
            vad_settings,
            variable=self._vad_threshold_var,
            from_=0.001, to=0.05, resolution=0.001,
            orient=tk.HORIZONTAL, length=130,
            bg=PANEL_COLOR, fg=TEXT_COLOR,
            troughcolor="#444466", highlightthickness=0,
            command=self._on_threshold_change,
        ).grid(row=0, column=1, padx=(4, 0))

        tk.Label(vad_settings, text="Silence timeout (s):", font=FONT_SMALL,
                 bg=PANEL_COLOR, fg=SUBTEXT_COLOR).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self._vad_silence_var = tk.DoubleVar(value=config.VAD_SILENCE_DURATION)
        tk.Scale(
            vad_settings,
            variable=self._vad_silence_var,
            from_=0.5, to=4.0, resolution=0.5,
            orient=tk.HORIZONTAL, length=130,
            bg=PANEL_COLOR, fg=TEXT_COLOR,
            troughcolor="#444466", highlightthickness=0,
            command=self._on_silence_change,
        ).grid(row=1, column=1, padx=(4, 0), pady=(4, 0))

        # ── Recording buttons ─────────────────────────────────────────────────
        btn_frame = tk.Frame(right, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, pady=(0, 6))

        # Auto-listen (VAD) toggle
        self._vad_btn = tk.Button(
            btn_frame,
            text="🤖  AUTO LISTEN (VAD)",
            font=("Segoe UI", 10, "bold"),
            bg=VAD_COLOR, fg="white",
            activebackground="#0284c7",
            relief=tk.FLAT, padx=8, pady=6,
            command=self._toggle_vad,
            cursor="hand2",
        )
        self._vad_btn.pack(fill=tk.X, pady=(0, 4))

        # Manual record
        self._rec_btn = tk.Button(
            btn_frame,
            text="🎙  HOLD TO RECORD (Manual)",
            font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=TEXT_COLOR,
            activebackground="#4b5563",
            relief=tk.FLAT, padx=8, pady=6,
            command=self._toggle_recording,
            cursor="hand2",
        )
        self._rec_btn.pack(fill=tk.X)

        # Status indicator
        self._rec_status_var = tk.StringVar(value="⬤  Idle")
        tk.Label(
            btn_frame,
            textvariable=self._rec_status_var,
            font=FONT_MAIN, bg=BG_COLOR, fg=SUBTEXT_COLOR,
        ).pack(anchor=tk.W, pady=(4, 0))

        # ── Transcript area ───────────────────────────────────────────────────
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

        btn_row = tk.Frame(right, bg=BG_COLOR)
        btn_row.pack(fill=tk.X, pady=(4, 0))

        tk.Button(
            btn_row, text="Clear",
            font=FONT_SMALL, bg=PANEL_COLOR, fg=SUBTEXT_COLOR,
            relief=tk.FLAT, command=self._clear_transcript, cursor="hand2",
        ).pack(side=tk.RIGHT)

        tk.Button(
            btn_row, text="📂 Open outputs/",
            font=FONT_SMALL, bg=PANEL_COLOR, fg=SUBTEXT_COLOR,
            relief=tk.FLAT, command=self._open_outputs, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        # ── Audio notification bar ────────────────────────────────────────────
        # Shown/hidden dynamically after each translation audio is saved
        self._audio_notify_frame = tk.Frame(right, bg="#1a2e1a", bd=1, relief=tk.SOLID)
        # (packed dynamically in _show_audio_notification)

        self._audio_file_var = tk.StringVar(value="")
        tk.Label(
            self._audio_notify_frame,
            text="🔊 Audio ready:",
            font=FONT_SMALL, bg="#1a2e1a", fg=SUCCESS_COLOR,
        ).pack(side=tk.LEFT, padx=(8, 4), pady=6)

        self._audio_name_label = tk.Label(
            self._audio_notify_frame,
            textvariable=self._audio_file_var,
            font=("Consolas", 9), bg="#1a2e1a", fg=TEXT_COLOR,
        )
        self._audio_name_label.pack(side=tk.LEFT, pady=6)

        tk.Button(
            self._audio_notify_frame,
            text="▶ Play",
            font=FONT_SMALL, bg=SUCCESS_COLOR, fg="white",
            relief=tk.FLAT, padx=8, pady=2,
            command=self._play_last_audio, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4, pady=4)

        tk.Button(
            self._audio_notify_frame,
            text="📂 Open",
            font=FONT_SMALL, bg=PANEL_COLOR, fg=SUBTEXT_COLOR,
            relief=tk.FLAT, padx=6, pady=2,
            command=self._open_outputs, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 2), pady=4)

        tk.Button(
            self._audio_notify_frame,
            text="✕",
            font=FONT_SMALL, bg="#1a2e1a", fg=SUBTEXT_COLOR,
            relief=tk.FLAT, padx=4, pady=2,
            command=self._hide_audio_notification, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=2, pady=4)

        # Status bar
        self._status_var = tk.StringVar(value="⏳  Loading Whisper model…")
        tk.Label(
            right,
            textvariable=self._status_var,
            font=FONT_MAIN, bg=BG_COLOR, fg=SUBTEXT_COLOR,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(6, 0))

    # ══════════════════════════════════════════════════════════════════════════
    #  Webcam Feed
    # ══════════════════════════════════════════════════════════════════════════

    def _start_webcam(self):
        self._face_analyzer.start()
        self._webcam_running = True
        self._update_webcam()

    def _update_webcam(self):
        if not self._webcam_running:
            return
        frame, results = self._face_analyzer.get_latest()

        if frame is not None:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((480, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            self._cam_label.configure(image=photo)
            self._cam_label.image = photo

            if results:
                parts = []
                for r in results:
                    gender = r["gender"]
                    age    = r["age"]       # already formatted: "~27y" or "(25-32)" or "?"
                    conf   = r.get("confidence", 0)
                    parts.append(f"{gender}  {age}  ({conf:.0f}% conf)")
                self._face_info_var.set("  |  ".join(parts))
            else:
                self._face_info_var.set("No face detected")

        self.after(33, self._update_webcam)

    # ══════════════════════════════════════════════════════════════════════════
    #  Mic Energy Meter
    # ══════════════════════════════════════════════════════════════════════════

    def _start_energy_meter(self):
        """Sample mic energy every 100ms and update the progress bar."""
        self._energy_stream = None
        self._energy_rms    = 0.0
        self._energy_running = True

        def _read():
            try:
                with sd.InputStream(
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=1, dtype="float32",
                    blocksize=int(config.AUDIO_SAMPLE_RATE * 0.1),
                ) as stream:
                    while self._energy_running:
                        block, _ = stream.read(int(config.AUDIO_SAMPLE_RATE * 0.1))
                        self._energy_rms = float(np.sqrt(np.mean(block ** 2)))
            except Exception:
                pass

        threading.Thread(target=_read, daemon=True).start()
        self._refresh_energy()

    def _refresh_energy(self):
        rms = self._energy_rms
        # Scale 0–0.1 to 0–100
        pct = min(rms * 1000, 100)
        self._energy_bar["value"] = pct
        self._energy_var.set(f"{rms:.3f}")

        # Color threshold indicator
        threshold = self._vad_threshold_var.get()
        self._energy_bar.configure(
            style="Green.Horizontal.TProgressbar"
            if rms > threshold else "TProgressbar"
        )
        self.after(100, self._refresh_energy)

    # ══════════════════════════════════════════════════════════════════════════
    #  Whisper Model Loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_whisper_async(self):
        src_lang = config.SUPPORTED_LANGUAGES.get(
            self._src_lang_var.get(), config.DEFAULT_SOURCE_LANG
        )
        tgt_lang = config.SUPPORTED_LANGUAGES.get(
            self._tgt_lang_var.get(), config.DEFAULT_TARGET_LANG
        )
        self._speech_handler = SpeechHandler(source_lang=src_lang)
        self._translator     = Translator(source=src_lang, target=tgt_lang)

        def _load():
            self._speech_handler.load_model(
                on_progress=lambda msg: self.after(0, self._set_status, msg)
            )
            self._model_loaded = True
            self.after(0, self._set_status, "✅  Ready — Whisper loaded")

        threading.Thread(target=_load, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  VAD — Auto Listen Mode
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_vad(self):
        if self._vad_on:
            self._stop_vad()
        else:
            self._start_vad()

    def _start_vad(self):
        if not self._model_loaded:
            messagebox.showwarning("Not Ready", "Whisper model is still loading.")
            return
        if self._recording:
            messagebox.showwarning("Busy", "Stop manual recording first.")
            return
        self._vad_on = True
        config.VAD_ENERGY_THRESHOLD = self._vad_threshold_var.get()
        config.VAD_SILENCE_DURATION = self._vad_silence_var.get()
        self._vad_btn.configure(
            text="⏹  STOP AUTO LISTEN",
            bg="#b91c1c", activebackground="#991b1b",
        )
        self._rec_status_var.set("👂  Listening for speech…")
        self._set_status("VAD active — speak now")
        self._speech_handler.start_vad(on_transcript=self._on_vad_transcript)

    def _stop_vad(self):
        self._vad_on = False
        self._speech_handler.stop_vad()
        self._vad_btn.configure(
            text="🤖  AUTO LISTEN (VAD)", bg=VAD_COLOR,
            activebackground="#0284c7",
        )
        self._rec_status_var.set("⬤  Idle")
        self._set_status("VAD stopped")

    def _on_vad_transcript(self, original: str):
        """Called from VAD background thread when speech segment is transcribed."""
        self.after(0, self._rec_status_var.set, "⚙️  Translating…")
        translated = self._translator.translate(original) if self._translator else original
        src_code   = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), "te")
        tgt_code   = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), "hi")
        self.after(0, self._handle_output, original, translated, src_code, tgt_code)
        self.after(0, self._rec_status_var.set, "👂  Listening for speech…")

    # ══════════════════════════════════════════════════════════════════════════
    #  Manual Push-to-Record
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self._model_loaded:
            messagebox.showwarning("Not Ready", "Whisper model is still loading.")
            return
        if self._vad_on:
            messagebox.showwarning("Busy", "Stop Auto Listen first.")
            return
        self._recording = True
        self._rec_btn.configure(
            text="⏹  CLICK TO STOP", bg=REC_COLOR,
            activebackground="#b91c1c",
        )
        self._rec_status_var.set("🔴  Recording…")
        self._set_status("Recording — speak now")
        self._speech_handler.start_recording()

    def _stop_recording(self):
        self._recording = False
        self._rec_btn.configure(
            text="🎙  HOLD TO RECORD (Manual)", bg="#374151",
            activebackground="#4b5563",
        )
        self._rec_status_var.set("⚙️  Processing…")
        self._set_status("Transcribing…")
        threading.Thread(target=self._process_manual_recording, daemon=True).start()

    def _process_manual_recording(self):
        original = self._speech_handler.stop_recording()
        if not original:
            self.after(0, self._set_status, "⚠️  No speech detected")
            self.after(0, self._rec_status_var.set, "⬤  Idle")
            return
        self.after(0, self._set_status, "Translating…")
        translated = self._translator.translate(original) if self._translator else original
        src_code   = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), "te")
        tgt_code   = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), "hi")
        self.after(0, self._handle_output, original, translated, src_code, tgt_code)
        self.after(0, self._rec_status_var.set, "⬤  Idle")

    # ── Shared output handler ─────────────────────────────────────────────────

    def _handle_output(self, original, translated, src_code, tgt_code):
        """Display transcript in UI and save files based on output mode."""
        self._append_transcript(original, translated, src_code, tgt_code)
        mode = self._output_mode_var.get()

        if mode in ("transcript", "both"):
            saved = self._tts_handler.save_transcript(
                original, translated, src_code, tgt_code
            )
            self._set_status(f"✅  Transcript saved → {os.path.basename(saved)}")

        if mode in ("audio", "both"):
            # Save audio file — do NOT auto-play; show notification bar instead
            audio_path = self._tts_handler.speak_and_save(
                translated, lang=tgt_code, play=False
            )
            # Show the audio-ready notification bar so user can choose to play
            self.after(800, self._show_audio_notification, audio_path)

    # ── Audio Notification Bar ─────────────────────────────────────────────────

    def _show_audio_notification(self, filepath: str):
        """Show the green 'Audio ready' bar with Play / Open / Dismiss buttons."""
        self._last_audio_path = filepath
        self._audio_file_var.set(os.path.basename(filepath))
        self._audio_notify_frame.pack(fill=tk.X, pady=(4, 0))
        self._set_status(f"🔊 Audio saved — click ▶ Play when ready")

    def _hide_audio_notification(self):
        """Dismiss the audio notification bar."""
        self._audio_notify_frame.pack_forget()

    def _play_last_audio(self):
        """Play the last saved audio file via pyttsx3/pygame."""
        path = getattr(self, "_last_audio_path", None)
        if not path or not os.path.exists(path):
            messagebox.showwarning("Not Found", "Audio file not found.")
            return
        # Play in background so UI stays responsive
        import threading
        import pygame
        def _play():
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(5)
            except Exception as e:
                print(f"[Playback] Error: {e}")
        threading.Thread(target=_play, daemon=True).start()
        self._set_status(f"▶ Playing: {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Language Settings
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_language_settings(self):
        src_name = self._src_lang_var.get()
        tgt_name = self._tgt_lang_var.get()
        src_code = config.SUPPORTED_LANGUAGES.get(src_name, "en")
        tgt_code = config.SUPPORTED_LANGUAGES.get(tgt_name, "en")

        if src_code == tgt_code:
            messagebox.showwarning("Same Language",
                                   "Source and target must be different.")
            return
        if self._speech_handler:
            self._speech_handler.source_lang = src_code
        if self._translator:
            self._translator.update_languages(src_code, tgt_code)
        self._set_status(f"✅  Languages: {src_name} → {tgt_name}")

    def _on_threshold_change(self, _=None):
        config.VAD_ENERGY_THRESHOLD = self._vad_threshold_var.get()

    def _on_silence_change(self, _=None):
        config.VAD_SILENCE_DURATION = self._vad_silence_var.get()

    # ══════════════════════════════════════════════════════════════════════════
    #  Transcript Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _append_transcript(self, original, translated, src_code, tgt_code):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._transcript_box.configure(state=tk.NORMAL)
        self._transcript_box.insert(tk.END,
            f"\n[{ts}]\n"
            f"🎤 ({src_code}): {original}\n"
            f"🌐 ({tgt_code}): {translated}\n"
            f"{'─' * 42}\n"
        )
        self._transcript_box.configure(state=tk.DISABLED)
        self._transcript_box.see(tk.END)

    def _clear_transcript(self):
        self._transcript_box.configure(state=tk.NORMAL)
        self._transcript_box.delete("1.0", tk.END)
        self._transcript_box.configure(state=tk.DISABLED)

    def _open_outputs(self):
        import subprocess
        subprocess.Popen(f'explorer "{os.path.abspath("outputs")}"')

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    # ══════════════════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._webcam_running  = False
        self._energy_running  = False
        self._face_analyzer.stop()
        if self._vad_on:
            self._speech_handler.stop_vad()
        if self._recording:
            self._speech_handler.stop_recording()
        self._tts_handler.stop_playback()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
