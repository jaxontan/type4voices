@echo off
setlocal
title Voice Dictation (Local Qwen3-ASR)
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Python virtual environment not found.
    echo.
    echo Run setup.bat once to create it, then start this file again.
    echo Expected: "%~dp0.venv\Scripts\python.exe"
    echo.
    pause
    exit /b 1
)

echo Starting Voice Dictation from "%~dp0"...
echo Logs: "%~dp0dictate.log"
echo.
"%PYTHON%" -u "%~dp0dictate.py"
echo.
echo Voice Dictation exited with code %ERRORLEVEL%.
pause
