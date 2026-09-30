@echo off
setlocal

cd /d "%~dp0"
title Lumen Launcher

set "LUMEN_VENV=.venv"
set "LUMEN_PYTHON=%LUMEN_VENV%\Scripts\python.exe"

if not exist "%LUMEN_PYTHON%" (
    echo [LUMEN] Preparing Python environment...

    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        py -3.12 -m venv "%LUMEN_VENV%"
    ) else (
        python --version >nul 2>&1
        if errorlevel 1 goto :python_missing
        python -m venv "%LUMEN_VENV%"
    )

    if errorlevel 1 goto :venv_failed
)

"%LUMEN_PYTHON%" -c "import cv2, dotenv, glfw, httpx, mediapipe, moderngl, numpy, omegaconf, pocketsphinx, sounddevice, soundfile, torch, torchaudio, trimesh, vosk, PIL" >nul 2>&1
if errorlevel 1 (
    echo [LUMEN] Installing required packages. This is needed only once...
    "%LUMEN_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :install_failed
)

echo [LUMEN] Starting...
"%LUMEN_PYTHON%" main.py
set "LUMEN_EXIT_CODE=%ERRORLEVEL%"

if not "%LUMEN_EXIT_CODE%"=="0" (
    echo.
    echo [LUMEN] The application closed with an error.
    pause
)

exit /b %LUMEN_EXIT_CODE%

:python_missing
echo.
echo [LUMEN] Python was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
pause
exit /b 1

:venv_failed
echo.
echo [LUMEN] Could not create the local Python environment.
pause
exit /b 1

:install_failed
echo.
echo [LUMEN] Package installation failed.
echo Check the internet connection, then run this file again.
pause
exit /b 1
