"""
app.py  v3  -- CustomTkinter Modern UI
Upgrades over v2:
  - CustomTkinter dark theme, rounded corners, modern widgets
  - FPS counter live on the webcam header
  - Face info badges drawn directly on the video frame
  - Color-coded transcript (original amber, translation green)
  - Segmented output-mode selector
  - Toast-style audio notification
  - All original functionality preserved (VAD, manual record, TTS, etc.)
"""

import os
import time
import threading
import datetime
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sounddevice as sd

from modules.face_analyzer  import FaceAnalyzer
from modules.speech_handler import SpeechHandler
from modules.translator     import Translator
from modules.tts_handler    import TTSHandler
import config

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG            = "#0f0f1a"
PANEL         = "#1a1a2e"
PANEL2        = "#16213e"
ACCENT        = "#7c3aed"
ACCENT2       = "#0ea5e9"
REC_COLOR     = "#dc2626"
SUCCESS       = "#16a34a"
WARN          = "#f59e0b"
TEXT          = "#e2e8f0"
SUBTEXT       = "#64748b"
TRANS_COLOR   = "#34d399"
ORIG_COLOR    = "#fbbf24"

FONT_BOLD  = ("Segoe UI", 12, "bold")
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 11)
FONT_TITLE = ("Segoe UI", 15, "bold")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Detector N Translator  v3")
        self.geometry("1200x720")
        self.minsize(900, 600)
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._face_analyzer  = FaceAnalyzer()
        self._speech_handler = None
        self._translator     = None
        self._tts_handler    = TTSHandler()

        self._webcam_running  = False
        self._model_loaded    = False
        self._recording       = False
        self._vad_on          = False
        self._energy_running  = False
        self._energy_rms      = 0.0
        self._last_audio_path = None
        self._fps_times       = []
        self._fps             = 0.0

        self._build_ui()
        self._start_webcam()
        self._load_whisper_async()
        self._start_energy_meter()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=ACCENT, corner_radius=0, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="Detector N Translator",
            font=("Segoe UI", 14, "bold"), text_color="white",
        ).pack(side="left", padx=16, pady=8)

        ctk.CTkLabel(
            header, text="v3  |  InsightFace + Whisper + CustomTkinter",
            font=("Segoe UI", 10), text_color="#c4b5fd",
        ).pack(side="left")

        self._fps_label = ctk.CTkLabel(
            header, text="FPS: --",
            font=("Segoe UI", 10), text_color="#c4b5fd",
        )
        self._fps_label.pack(side="right", padx=16)

        # Main area
        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(fill="both", expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_panel(main)

        # Status bar
        self._status_var = tk.StringVar(value="Loading Whisper model...")
        status_bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=28)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        ctk.CTkLabel(
            status_bar, textvariable=self._status_var,
            font=("Segoe UI", 10), text_color=SUBTEXT, anchor="w",
        ).pack(side="left", padx=12)

    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(left, fg_color=PANEL2, corner_radius=8)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        ctk.CTkLabel(hdr, text="Live Feed", font=FONT_BOLD, text_color=TEXT,
                     ).pack(side="left", padx=10, pady=6)

        self._face_badge = ctk.CTkLabel(
            hdr, text="No face detected",
            font=FONT_SMALL, text_color=SUBTEXT,
        )
        self._face_badge.pack(side="right", padx=10, pady=6)

        self._cam_label = ctk.CTkLabel(left, text="", fg_color="#000000", corner_radius=8)
        self._cam_label.grid(row=1, column=0, sticky="nsew", padx=10)

        mic_row = ctk.CTkFrame(left, fg_color=PANEL)
        mic_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 10))

        ctk.CTkLabel(mic_row, text="Mic:", font=FONT_SMALL, text_color=SUBTEXT,
                     ).pack(side="left", padx=(8, 4))

        self._energy_bar = ctk.CTkProgressBar(mic_row, width=260, height=10, progress_color=ACCENT2)
        self._energy_bar.set(0)
        self._energy_bar.pack(side="left", padx=4)

        self._energy_val = ctk.CTkLabel(mic_row, text="0.000", font=("Consolas", 9),
                                        text_color=SUBTEXT, width=50)
        self._energy_val.pack(side="left")

    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(right, fg_color=PANEL, corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=2, pady=2)
        scroll.columnconfigure(0, weight=1)
        row = 0

        # Language settings
        lang_card = self._card(scroll, "Language Settings")
        lang_card.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4)); row += 1

        lang_names = list(config.SUPPORTED_LANGUAGES.keys())
        self._src_lang_var = tk.StringVar(value="Telugu")
        self._tgt_lang_var = tk.StringVar(value="Hindi")
        self._add_label_combo(lang_card, "Speaker (Source):", self._src_lang_var, lang_names, 0)
        self._add_label_combo(lang_card, "Listener (Target):", self._tgt_lang_var, lang_names, 1)
        ctk.CTkButton(lang_card, text="Apply Languages", font=FONT_BOLD, height=32,
                      fg_color=ACCENT, hover_color="#6d28d9",
                      command=self._apply_language_settings,
                      ).grid(row=3, column=0, columnspan=2, sticky="e", padx=8, pady=(4, 8))

        # Output mode
        out_card = self._card(scroll, "Output Mode")
        out_card.grid(row=row, column=0, sticky="ew", padx=8, pady=4); row += 1

        self._output_mode_var = tk.StringVar(value="Both")
        self._seg_btn = ctk.CTkSegmentedButton(
            out_card, values=["Transcript", "Audio only", "Both"],
            variable=self._output_mode_var,
            font=FONT_SMALL,
            selected_color=ACCENT, selected_hover_color="#6d28d9",
            unselected_color=PANEL2,
        )
        self._seg_btn.grid(row=1, column=0, padx=8, pady=8, sticky="ew")

        # VAD settings
        vad_card = self._card(scroll, "Voice Activity Detection")
        vad_card.grid(row=row, column=0, sticky="ew", padx=8, pady=4); row += 1
        vad_card.columnconfigure(1, weight=1)

        self._vad_threshold_var = tk.DoubleVar(value=config.VAD_ENERGY_THRESHOLD)
        self._vad_silence_var   = tk.DoubleVar(value=config.VAD_SILENCE_DURATION)

        ctk.CTkLabel(vad_card, text="Sensitivity:", font=FONT_SMALL, text_color=SUBTEXT,
                     ).grid(row=1, column=0, sticky="w", padx=8, pady=(4, 2))
        ctk.CTkSlider(vad_card, from_=0.001, to=0.05, variable=self._vad_threshold_var,
                      progress_color=ACCENT2, command=self._on_threshold_change,
                      ).grid(row=1, column=1, sticky="ew", padx=8, pady=(4, 2))

        ctk.CTkLabel(vad_card, text="Silence (s):", font=FONT_SMALL, text_color=SUBTEXT,
                     ).grid(row=2, column=0, sticky="w", padx=8, pady=(2, 8))
        ctk.CTkSlider(vad_card, from_=0.5, to=4.0, variable=self._vad_silence_var,
                      progress_color=ACCENT2, command=self._on_silence_change,
                      ).grid(row=2, column=1, sticky="ew", padx=8, pady=(2, 8))

        # Recording buttons
        btn_card = self._card(scroll, "Recording Controls")
        btn_card.grid(row=row, column=0, sticky="ew", padx=8, pady=4); row += 1

        self._vad_btn = ctk.CTkButton(
            btn_card, text="AUTO LISTEN  (VAD)", font=FONT_BOLD, height=40,
            fg_color=ACCENT2, hover_color="#0284c7", command=self._toggle_vad,
        )
        self._vad_btn.pack(fill="x", padx=8, pady=(8, 4))

        self._rec_btn = ctk.CTkButton(
            btn_card, text="HOLD TO RECORD  (Manual)", font=FONT_BOLD, height=38,
            fg_color=PANEL2, hover_color="#2d3748", command=self._toggle_recording,
        )
        self._rec_btn.pack(fill="x", padx=8, pady=(0, 4))

        self._rec_status_var = tk.StringVar(value="Idle")
        ctk.CTkLabel(btn_card, textvariable=self._rec_status_var,
                     font=FONT_SMALL, text_color=SUBTEXT,
                     ).pack(anchor="w", padx=12, pady=(0, 8))

        # Transcript
        tx_card = ctk.CTkFrame(scroll, fg_color=PANEL2, corner_radius=10)
        tx_card.grid(row=row, column=0, sticky="ew", padx=8, pady=4); row += 1
        tx_card.columnconfigure(0, weight=1)

        tx_hdr = ctk.CTkFrame(tx_card, fg_color=PANEL2)
        tx_hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        ctk.CTkLabel(tx_hdr, text="Transcript / Translation", font=FONT_BOLD, text_color=TEXT,
                     ).pack(side="left")
        ctk.CTkButton(tx_hdr, text="Clear", width=60, height=26, font=FONT_SMALL,
                      fg_color=PANEL, hover_color="#374151", command=self._clear_transcript,
                      ).pack(side="right")
        ctk.CTkButton(tx_hdr, text="Open Folder", width=90, height=26, font=FONT_SMALL,
                      fg_color=PANEL, hover_color="#374151", command=self._open_outputs,
                      ).pack(side="right", padx=(0, 4))

        self._transcript_box = ctk.CTkTextbox(
            tx_card, height=220, font=FONT_MONO,
            fg_color=PANEL, text_color=TEXT, corner_radius=8,
            wrap="word", state="disabled",
        )
        self._transcript_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        self._transcript_box._textbox.tag_configure("orig",  foreground=ORIG_COLOR)
        self._transcript_box._textbox.tag_configure("trans", foreground=TRANS_COLOR)
        self._transcript_box._textbox.tag_configure("meta",  foreground=SUBTEXT)
        self._transcript_box._textbox.tag_configure("sep",   foreground="#2d3748")

        # Audio toast
        self._audio_toast = ctk.CTkFrame(scroll, fg_color="#1a2e1a", corner_radius=8)
        self._audio_file_var = tk.StringVar(value="")
        ctk.CTkLabel(self._audio_toast, text="Audio ready:", font=FONT_SMALL,
                     text_color=TRANS_COLOR).pack(side="left", padx=(10, 4), pady=6)
        ctk.CTkLabel(self._audio_toast, textvariable=self._audio_file_var,
                     font=("Consolas", 9), text_color=TEXT).pack(side="left", pady=6)
        ctk.CTkButton(self._audio_toast, text="Replay", width=72, height=26,
                      font=FONT_SMALL, fg_color=SUCCESS, hover_color="#15803d",
                      command=self._play_last_audio).pack(side="right", padx=4, pady=4)
        ctk.CTkButton(self._audio_toast, text="X", width=28, height=26,
                      font=FONT_SMALL, fg_color=PANEL2, hover_color="#374151",
                      command=self._hide_audio_notification).pack(side="right", padx=(0, 2), pady=4)
        self._toast_row = row

    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=PANEL2, corner_radius=10)
        frame.columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=title, font=FONT_BOLD, text_color=TEXT, anchor="w",
                     ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 2))
        return frame

    def _add_label_combo(self, parent, label, var, values, row_idx):
        ctk.CTkLabel(parent, text=label, font=FONT_SMALL, text_color=SUBTEXT,
                     ).grid(row=row_idx + 1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkComboBox(parent, variable=var, values=values, state="readonly", width=160,
                        button_color=ACCENT, border_color=ACCENT, dropdown_hover_color=ACCENT,
                        ).grid(row=row_idx + 1, column=1, padx=(4, 10), pady=2, sticky="w")

    def _start_webcam(self):
        self._face_analyzer.start()
        self._webcam_running = True
        self._update_webcam()

    def _update_webcam(self):
        if not self._webcam_running:
            return

        now = time.time()
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        self._fps_times.append(now)
        self._fps = len(self._fps_times)

        frame, results = self._face_analyzer.get_latest()

        if frame is not None:
            display = frame.copy()
            if results:
                face_texts = []
                for r in results:
                    x, y, w, h = r["box"]
                    age, gender = r["age"], r["gender"]
                    conf = r.get("confidence", 0)
                    label = f"{gender}  {age}  {conf:.0f}%"
                    face_texts.append(f"{gender} {age} ({conf:.0f}%)")
                    cv2.rectangle(display, (x, y), (x + w, y + h), (124, 58, 237), 2)
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(display, (x, y - lh - 12), (x + lw + 8, y), (124, 58, 237), -1)
                    cv2.putText(display, label, (x + 4, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                self._face_badge.configure(text="  |  ".join(face_texts), text_color=TEXT)
            else:
                status = self._face_analyzer.status
                msg = "Loading models..." if "Loading" in status else "No face detected"
                color = WARN if "Loading" in status else SUBTEXT
                self._face_badge.configure(text=msg, text_color=color)

            cv2.putText(display, f"FPS: {self._fps}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 211, 153), 1, cv2.LINE_AA)

            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((580, 435), Image.LANCZOS)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(580, 435))
            self._cam_label.configure(image=photo, text="")
            self._cam_label.image = photo

        self._fps_label.configure(text=f"FPS: {self._fps}")
        self.after(33, self._update_webcam)

    def _start_energy_meter(self):
        self._energy_running = True

        def _read():
            try:
                with sd.InputStream(samplerate=config.AUDIO_SAMPLE_RATE,
                                    channels=1, dtype="float32",
                                    blocksize=int(config.AUDIO_SAMPLE_RATE * 0.1)) as stream:
                    while self._energy_running:
                        block, _ = stream.read(int(config.AUDIO_SAMPLE_RATE * 0.1))
                        self._energy_rms = float(np.sqrt(np.mean(block ** 2)))
            except Exception:
                pass

        threading.Thread(target=_read, daemon=True).start()
        self._refresh_energy()

    def _refresh_energy(self):
        rms = self._energy_rms
        self._energy_bar.set(min(rms / 0.1, 1.0))
        self._energy_val.configure(text=f"{rms:.3f}")
        color = SUCCESS if rms > self._vad_threshold_var.get() else ACCENT2
        self._energy_bar.configure(progress_color=color)
        self.after(100, self._refresh_energy)

    def _load_whisper_async(self):
        src = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), config.DEFAULT_SOURCE_LANG)
        tgt = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), config.DEFAULT_TARGET_LANG)
        self._speech_handler = SpeechHandler(source_lang=src)
        self._translator     = Translator(source=src, target=tgt)

        def _load():
            self._speech_handler.load_model(
                on_progress=lambda msg: self.after(0, self._set_status, msg))
            self._model_loaded = True
            self.after(0, self._set_status, "Ready -- Whisper loaded")

        threading.Thread(target=_load, daemon=True).start()

    def _toggle_vad(self):
        self._stop_vad() if self._vad_on else self._start_vad()

    def _start_vad(self):
        if not self._model_loaded:
            self._set_status("Whisper still loading -- please wait..."); return
        if self._recording:
            self._set_status("Stop manual recording first."); return
        self._vad_on = True
        config.VAD_ENERGY_THRESHOLD = self._vad_threshold_var.get()
        config.VAD_SILENCE_DURATION = self._vad_silence_var.get()
        self._vad_btn.configure(text="STOP AUTO LISTEN", fg_color=REC_COLOR, hover_color="#b91c1c")
        self._rec_status_var.set("Listening for speech...")
        self._set_status("VAD active -- speak now")
        self._speech_handler.start_vad(on_transcript=self._on_vad_transcript)

    def _stop_vad(self):
        self._vad_on = False
        self._speech_handler.stop_vad()
        self._vad_btn.configure(text="AUTO LISTEN  (VAD)", fg_color=ACCENT2, hover_color="#0284c7")
        self._rec_status_var.set("Idle")
        self._set_status("VAD stopped")

    def _on_vad_transcript(self, original: str):
        self.after(0, self._rec_status_var.set, "Translating...")
        translated = self._translator.translate(original) if self._translator else original
        src_code = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), "te")
        tgt_code = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), "hi")
        self.after(0, self._handle_output, original, translated, src_code, tgt_code)
        self.after(0, self._rec_status_var.set, "Listening for speech...")

    def _toggle_recording(self):
        self._stop_recording() if self._recording else self._start_recording()

    def _start_recording(self):
        if not self._model_loaded:
            self._set_status("Whisper still loading -- please wait..."); return
        if self._vad_on:
            self._set_status("Stop Auto Listen first."); return
        self._recording = True
        self._rec_btn.configure(text="CLICK TO STOP", fg_color=REC_COLOR, hover_color="#b91c1c")
        self._rec_status_var.set("Recording...")
        self._set_status("Recording -- speak now")
        self._speech_handler.start_recording()

    def _stop_recording(self):
        self._recording = False
        self._rec_btn.configure(text="HOLD TO RECORD  (Manual)", fg_color=PANEL2, hover_color="#2d3748")
        self._rec_status_var.set("Processing...")
        self._set_status("Transcribing...")
        threading.Thread(target=self._process_manual_recording, daemon=True).start()

    def _process_manual_recording(self):
        original = self._speech_handler.stop_recording()
        if not original:
            self.after(0, self._set_status, "No speech detected")
            self.after(0, self._rec_status_var.set, "Idle"); return
        self.after(0, self._set_status, "Translating...")
        translated = self._translator.translate(original) if self._translator else original
        src_code = config.SUPPORTED_LANGUAGES.get(self._src_lang_var.get(), "te")
        tgt_code = config.SUPPORTED_LANGUAGES.get(self._tgt_lang_var.get(), "hi")
        self.after(0, self._handle_output, original, translated, src_code, tgt_code)
        self.after(0, self._rec_status_var.set, "Idle")

    def _handle_output(self, original, translated, src_code, tgt_code):
        self._append_transcript(original, translated, src_code, tgt_code)
        mode_map = {"Transcript": "transcript", "Audio only": "audio", "Both": "both"}
        mode = mode_map.get(self._output_mode_var.get(), "both")
        if mode in ("transcript", "both"):
            saved = self._tts_handler.save_transcript(original, translated, src_code, tgt_code)
            self._set_status(f"Transcript saved: {os.path.basename(saved)}")
        if mode in ("audio", "both"):
            audio_path = self._tts_handler.speak_and_save(translated, lang=tgt_code, play=True)
            if audio_path:
                self.after(200, self._show_audio_notification, audio_path)

    def _show_audio_notification(self, filepath: str):
        self._last_audio_path = filepath
        self._audio_file_var.set(os.path.basename(filepath))
        self._audio_toast.grid(row=self._toast_row, column=0, sticky="ew", padx=8, pady=(4, 8))
        self._set_status("Audio played + saved -- click Replay anytime")

    def _hide_audio_notification(self):
        self._audio_toast.grid_forget()

    def _play_last_audio(self):
        path = getattr(self, "_last_audio_path", None)
        if not path or not os.path.exists(path):
            self._set_status("Audio file not found."); return
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
        self._set_status(f"Playing: {os.path.basename(path)}")

    def _apply_language_settings(self):
        src_name = self._src_lang_var.get()
        tgt_name = self._tgt_lang_var.get()
        src_code = config.SUPPORTED_LANGUAGES.get(src_name, "en")
        tgt_code = config.SUPPORTED_LANGUAGES.get(tgt_name, "en")
        if src_code == tgt_code:
            self._set_status("Source and target must be different!"); return
        if self._speech_handler:
            self._speech_handler.source_lang = src_code
        if self._translator:
            self._translator.update_languages(src_code, tgt_code)
        self._set_status(f"Languages set: {src_name} -> {tgt_name}")

    def _on_threshold_change(self, _=None):
        config.VAD_ENERGY_THRESHOLD = self._vad_threshold_var.get()

    def _on_silence_change(self, _=None):
        config.VAD_SILENCE_DURATION = self._vad_silence_var.get()

    def _append_transcript(self, original, translated, src_code, tgt_code):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        box = self._transcript_box._textbox
        self._transcript_box.configure(state="normal")
        box.insert("end", f"[{ts}]\n", "meta")
        box.insert("end", f"({src_code}): ", "meta")
        box.insert("end", f"{original}\n", "orig")
        box.insert("end", f"({tgt_code}): ", "meta")
        box.insert("end", f"{translated}\n", "trans")
        box.insert("end", f"{'--'*19}\n", "sep")
        self._transcript_box.configure(state="disabled")
        self._transcript_box._textbox.see("end")

    def _clear_transcript(self):
        self._transcript_box.configure(state="normal")
        self._transcript_box.delete("0.0", "end")
        self._transcript_box.configure(state="disabled")

    def _open_outputs(self):
        import subprocess
        subprocess.Popen(f'explorer "{os.path.abspath("outputs")}"')

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _on_close(self):
        self._webcam_running = False
        self._energy_running = False
        self._face_analyzer.stop()
        if self._vad_on and self._speech_handler:
            self._speech_handler.stop_vad()
        if self._recording and self._speech_handler:
            self._speech_handler.stop_recording()
        self._tts_handler.stop_playback()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()