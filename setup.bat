@echo off
title Face Analysis + Speech Translator - SETUP
echo.
echo =====================================================
echo   Face Analysis + Speech Translator  v2
echo   Setup Script
echo =====================================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version

echo.
echo =====================================================
echo  Step 1/5 - Upgrading pip
echo =====================================================
python -m pip install --upgrade pip --quiet

echo.
echo =====================================================
echo  Step 2/5 - Installing core dependencies
echo  (this may take 3-5 minutes on first run)
echo =====================================================
pip install opencv-python deepface tf-keras --quiet
pip install openai-whisper sounddevice scipy --quiet
pip install deep-translator gtts pyttsx3 pygame Pillow numpy --quiet

echo.
echo =====================================================
echo  Step 3/5 - Installing InsightFace (BEST ACCURACY)
echo  Face detection + exact age/gender
echo =====================================================
pip install onnxruntime insightface --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] InsightFace install failed - app will use OpenCV fallback
) ELSE (
    echo [OK] InsightFace installed successfully
)

echo.
echo =====================================================
echo  Step 4/5 - Creating output directories
echo =====================================================
if not exist "outputs\audio"       mkdir "outputs\audio"
if not exist "outputs\transcripts" mkdir "outputs\transcripts"
if not exist "models"              mkdir "models"
echo [OK] Output directories created

echo.
echo =====================================================
echo  Step 5/5 - Verifying installation
echo =====================================================
python -c "
import sys
mods = {
    'cv2':              'opencv-python',
    'whisper':          'openai-whisper',
    'sounddevice':      'sounddevice',
    'deep_translator':  'deep-translator',
    'gtts':             'gTTS',
    'pyttsx3':          'pyttsx3',
    'pygame':           'pygame',
    'PIL':              'Pillow',
    'numpy':            'numpy',
}
ok, fail = [], []
for mod, pkg in mods.items():
    try:
        __import__(mod)
        ok.append(pkg)
    except ImportError:
        fail.append(pkg)

print(f'[OK]   Installed ({len(ok)}): {chr(44).join(ok)}')
if fail:
    print(f'[WARN] Missing  ({len(fail)}): {chr(44).join(fail)}')
else:
    print('[OK]   All core modules verified!')

try:
    import insightface, onnxruntime
    print('[OK]   InsightFace + ONNX Runtime ready (best accuracy)')
except ImportError:
    print('[INFO] InsightFace not available - will use OpenCV DNN fallback')
"

echo.
echo =====================================================
echo   Setup complete!
echo.
echo   NOTE: On FIRST RUN the app will auto-download:
echo     - Whisper 'small' model   (~460 MB, speech recognition)
echo     - InsightFace models      (~200 MB, face detection)
echo   These are one-time downloads, cached permanently.
echo.
echo   To start the app:
echo     python app.py
echo   Or double-click:
echo     run.bat
echo =====================================================
echo.
pause
