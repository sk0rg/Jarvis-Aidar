@echo off
chcp 65001 >nul
title JARVIS
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите setup_jarvis.bat для установки JARVIS.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" my_ai.py
if errorlevel 1 (
    echo.
    echo JARVIS завершился с ошибкой. Проверьте сообщения выше.
    pause
)
