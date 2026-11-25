@echo off
setlocal enabledelayedexpansion
title Voice Dictation - First-time Setup
cd /d "%~dp0.."

set "APP_DIR=%CD%"
set "PS_CHECK=%~dp0system_check.ps1"

echo ============================================================
echo  Voice Dictation - First-Time Setup and System Check
echo ============================================================

rem 1. Run Pre-flight System & Hardware Check
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_CHECK%"
if errorlevel 1 (
    echo [!] Pre-flight check encountered a warning, continuing setup...
)

rem 2. Find Python 3.10+ Executable
set "PYTHON_EXE="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_CHECK%" -Action find-python`) do (
    set "PYTHON_EXE=%%i"
)

if not defined PYTHON_EXE (
    echo.
    echo ============================================================
    echo  Python 3.10+ was not found on your system.
    echo ============================================================
    echo.
    set /p "INSTALL_PY=Would you like setup to install Python 3.12 automatically? (Y/N) [Y]: "
    if "!INSTALL_PY!"=="" set "INSTALL_PY=Y"
    if /i "!INSTALL_PY!"=="Y" (
        for /f "usebackq delims=" %%i in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_CHECK%" -Action install-python`) do (
            set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Python 3.12 is required but could not be located.
    echo Please install Python 3.12 manually from: https://www.python.org/downloads/
    echo Be sure to tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Using Python: "%PYTHON_EXE%"
echo.

rem 3. Virtual Environment (.venv)
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment (.venv) already exists, reusing it.
) else (
    echo Creating virtual environment in "%APP_DIR%\.venv" ...
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

rem 4. Hardware Accelerator & Dependencies Installation
echo.
echo ============================================================
echo  Installing Dependencies (this can take several minutes)...
echo ============================================================
".venv\Scripts\python.exe" -m pip install --upgrade pip

rem Check if Intel Arc GPU is present
set "IS_INTEL_ARC=0"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Arc|Intel\(R\) Graphics|Ultra' })) { Write-Output '1' } else { Write-Output '0' }"`) do (
    set "IS_INTEL_ARC=%%i"
)

if "%IS_INTEL_ARC%"=="1" (
    echo.
    echo [*] Intel Arc / Core Ultra GPU detected!
    echo     Ensuring Visual C++ 2015-2022 Redistributable is installed...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_CHECK%" -Action ensure-vcredist
    echo     Installing PyTorch with Intel Arc XPU hardware acceleration...
    ".venv\Scripts\python.exe" -m pip install "torch==2.15.0+xpu" --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
)

echo.
echo Installing remaining project requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Please check the error messages above.
    pause
    exit /b 1
)

rem 5. Environment Doctor Check
echo.
echo ============================================================
echo  Checking Environment Health...
echo ============================================================
".venv\Scripts\python.exe" dictate.py --doctor

rem 6. Taskbar / Traffic Light Tray Icon Tip
echo.
echo ============================================================
echo  Taskbar Traffic Light Icon Notice:
echo ============================================================
echo  On Windows, new tray icons are hidden by default inside
echo  the overflow menu behind the '^' chevron near the clock.
echo.
echo  If you do not see the green/red traffic light on your taskbar:
echo    1. Click the '^' chevron on the right side of your taskbar.
echo    2. Click and drag the traffic light icon onto the main taskbar.
echo.
echo  The app also automatically promotes the icon on future starts.
echo ============================================================

rem 7. Speech Model Download Check
set "CACHE_DIR=%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-ASR-1.7B"
if not exist "%CACHE_DIR%" (
    echo.
    echo ------------------------------------------------------------
    echo  Speech Recognition Model (~4.4 GB) is not yet downloaded.
    echo ------------------------------------------------------------
    set /p "DOWNLOAD_NOW=Would you like to download the model now? (Y/N) [Y]: "
    if "!DOWNLOAD_NOW!"=="" set "DOWNLOAD_NOW=Y"
    if /i "!DOWNLOAD_NOW!"=="Y" (
        echo Downloading speech model (this may take a few minutes)...
        ".venv\Scripts\python.exe" download_model.py
    ) else (
        echo You can download the model later by running:
        echo   .venv\Scripts\python.exe download_model.py
    )
) else (
    echo.
    echo [OK] Speech recognition model (Qwen3-ASR-1.7B) is already cached.
)

echo.
echo ============================================================
echo  Setup Complete!
echo  To start Voice Dictation, run: windows\run_dictate.bat
echo ============================================================
echo.
pause
