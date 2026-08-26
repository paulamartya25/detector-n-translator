@echo off
echo =====================================================
echo   Face Analysis + Speech Translator - SETUP
echo =====================================================
echo.

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo [3/4] Creating output directories...
if not exist "outputs\audio"       mkdir "outputs\audio"
if not exist "outputs\transcripts" mkdir "outputs\transcripts"

echo.
echo [4/4] Verifying installation...
python -c "import cv2, deepface, whisper, deep_translator, gtts, pygame, sounddevice; print('[OK] All modules imported successfully')"

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some modules may have failed to import. Check errors above.
) ELSE (
    echo.
    echo =====================================================
    echo   Setup complete! Run the app with:
    echo     python app.py
    echo =====================================================
)

pause
