"""
app_gradio.py -- HuggingFace Spaces deployment entry point
Mirrors all functionality of app.py (CustomTkinter) but uses
Gradio so it runs in a browser with no desktop install needed.
"""

import os, io, time, tempfile, threading
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# ── Module imports (same as desktop app) ──────────────────────
from modules.face_analyzer  import FaceAnalyzer
from modules.translator     import Translator
from modules.tts_handler    import TTSHandler
import config

# ── Global singletons ─────────────────────────────────────────
_face_analyzer = FaceAnalyzer()
_face_analyzer.start()

_translator   = Translator(source="te", target="hi")
_tts_handler  = TTSHandler()

LANG_NAMES = list(config.SUPPORTED_LANGUAGES.keys())

# ── Load Whisper lazily (heavy model) ────────────────────────
_whisper_model = None
_whisper_lock  = threading.Lock()

def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import whisper
            _whisper_model = whisper.load_model("small")
    return _whisper_model

# Pre-load in background so first call isn't slow
threading.Thread(target=_get_whisper, daemon=True).start()


# ── Face Analysis (image in, annotated image + info out) ─────
def analyze_face(image_np):
    """Called by Gradio webcam component on every frame."""
    if image_np is None:
        return None, "No image received"

    # Gradio sends RGB numpy — convert to BGR for OpenCV/InsightFace
    frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Push frame into the analyzer thread's queue
    _face_analyzer._latest_frame = frame_bgr

    # Get the latest result (may be from previous frame — that's fine)
    frame_out, results = _face_analyzer.get_latest()

    if frame_out is None:
        return image_np, "Loading face models (first run only)..."

    annotated = frame_out.copy()
    info_lines = []

    if results:
        for r in results:
            x, y, w, h = r["box"]
            age, gender = r["age"], r["gender"]
            conf = r.get("confidence", 0)
            label = f"{gender}  {age}  {conf:.0f}%"
            info_lines.append(f"{gender}, {age}, {conf:.0f}% confidence")

            cv2.rectangle(annotated, (x, y), (x+w, y+h), (124, 58, 237), 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y-lh-14), (x+lw+10, y), (124, 58, 237), -1)
            cv2.putText(annotated, label, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        face_info = "  |  ".join(info_lines)
    else:
        face_info = "No face detected"

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), face_info


# ── Speech → Translate → TTS ──────────────────────────────────
def transcribe_and_translate(audio_path, src_lang_name, tgt_lang_name):
    """Whisper transcription + translation + TTS."""
    if audio_path is None:
        return "No audio recorded", "", None

    src_code = config.SUPPORTED_LANGUAGES.get(src_lang_name, "te")
    tgt_code = config.SUPPORTED_LANGUAGES.get(tgt_lang_name, "hi")

    # Update translator languages
    _translator.update_languages(src_code, tgt_code)

    # Transcribe with Whisper
    try:
        model = _get_whisper()
        result = model.transcribe(audio_path, language=src_code, fp16=False)
        original = result["text"].strip()
    except Exception as e:
        return f"Transcription error: {e}", "", None

    if not original:
        return "No speech detected", "", None

    # Translate
    try:
        translated = _translator.translate(original)
    except Exception as e:
        translated = f"Translation error: {e}"

    # TTS
    try:
        audio_out = _tts_handler.speak_and_save(translated, lang=tgt_code, play=False)
    except Exception:
        audio_out = None

    return original, translated, audio_out


# ── Gradio UI ─────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0f0f1a",
    block_background_fill="#1a1a2e",
    block_label_text_color="#e2e8f0",
    block_title_text_color="#e2e8f0",
    input_background_fill="#16213e",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
)

CSS = """
.face-info { font-family: Consolas, monospace; color: #34d399; font-size: 14px; }
.header-title { font-size: 28px; font-weight: bold; color: #e2e8f0; }
.header-sub   { color: #64748b; font-size: 13px; }
footer { display: none !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="Detector N Translator") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div class="header-title">🎭 Detector N Translator</div>
        <div class="header-sub">
            Real-time Face Analysis (InsightFace + EfficientNetB3) &amp;
            Multilingual Speech Translation (Whisper + gTTS)
        </div>
    </div>
    """)

    with gr.Tab("Face Analysis"):
        gr.Markdown("### 📷 Live Webcam — Age & Gender Detection")
        gr.Markdown(
            "Allow camera access when prompted. The model detects faces and predicts "
            "**age** and **gender** in real time."
        )
        with gr.Row():
            with gr.Column(scale=2):
                cam_in  = gr.Image(source="webcam", streaming=True, label="Camera Feed", mirror_webcam=True)
                cam_out = gr.Image(label="Annotated Output")
            with gr.Column(scale=1):
                face_info = gr.Textbox(label="Detected Face Info", elem_classes="face-info",
                                       lines=4, interactive=False,
                                       value="Point your camera at a face...")
                gr.Markdown("""
                **Legend**
                - Box colour: purple = detected face
                - Label: Gender  Age  Confidence%

                **Model:** EfficientNetB3 custom-trained on UTKFace
                (Indian faces oversampled 3x for accuracy)
                """)

        cam_in.stream(fn=analyze_face, inputs=[cam_in], outputs=[cam_out, face_info])

    with gr.Tab("Speech Translation"):
        gr.Markdown("### 🎙️ Record Speech → Transcribe → Translate → Listen")
        gr.Markdown(
            "Select your **Source** language (what you'll speak) and "
            "**Target** language (what you want to hear). "
            "Then record your voice and press **Translate**."
        )
        with gr.Row():
            src_lang = gr.Dropdown(choices=LANG_NAMES, value="Telugu",  label="🗣️ Source Language (Speaker)")
            tgt_lang = gr.Dropdown(choices=LANG_NAMES, value="Hindi",   label="👂 Target Language (Listener)")

        audio_in = gr.Audio(source="microphone", type="filepath", label="🎤 Record your voice")

        translate_btn = gr.Button("🌐 Transcribe & Translate", variant="primary", size="lg")

        with gr.Row():
            original_out   = gr.Textbox(label="📝 Original Transcript", lines=3,
                                        interactive=False, elem_classes="face-info")
            translated_out = gr.Textbox(label="🌐 Translation", lines=3,
                                        interactive=False, elem_classes="face-info")

        audio_out = gr.Audio(label="🔊 Translated Audio", type="filepath", interactive=False)

        translate_btn.click(
            fn=transcribe_and_translate,
            inputs=[audio_in, src_lang, tgt_lang],
            outputs=[original_out, translated_out, audio_out],
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Project

        **Detector N Translator** is a real-time multimodal AI application that combines
        Computer Vision and Natural Language Processing.

        ### Tech Stack
        | Component | Technology |
        |---|---|
        | Face Detection | InsightFace SCRFD (buffalo_l) |
        | Age/Gender | Custom EfficientNetB3 (UTKFace, Indian-optimised) |
        | Speech Recognition | OpenAI Whisper (small, 244M params) |
        | Translation | Google Translate → MyMemory → Pons (3-level fallback) |
        | Text-to-Speech | gTTS + pyttsx3 |
        | Web UI | Gradio |
        | Desktop UI | CustomTkinter |

        ### Supported Languages
        Telugu, Hindi, English, Tamil, Kannada, Malayalam,
        Bengali, Marathi, Gujarati, Punjabi, Urdu

        ### GitHub
        [github.com/paulamartya25/detector-n-translator](https://github.com/paulamartya25/detector-n-translator)
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
