@echo off
setlocal
title Voice Dictation - Setup
cd /d "%~dp0.."

set "APP_DIR=%CD%"

echo ============================================================
echo  Voice Dictation - first-time setup
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" goto have_venv

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" not found.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

echo Creating virtual environment in "%APP_DIR%\.venv" ...
py -3.12 -m venv .venv
if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)
goto after_venv

:have_venv
echo Virtual environment already exists, reusing it.

:after_venv
echo.
echo Installing dependencies (this can take several minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo Checking the environment ...
".venv\Scripts\python.exe" dictate.py --doctor

echo.
echo ------------------------------------------------------------
echo The app needs the speech model downloaded once (~4.4 GB).
echo If the model cache check above failed, run:
echo     .venv\Scripts\python.exe download_model.py
echo ------------------------------------------------------------
echo.
pause
