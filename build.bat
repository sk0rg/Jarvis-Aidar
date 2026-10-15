@echo off
setlocal
cd /d "%~dp0"

if not exist "my_ai.py" (
    echo my_ai.py was not found.
    goto failed
)
if not exist "v4_ru.pt" (
    echo v4_ru.pt was not found next to this script.
    goto failed
)

if not exist ".venv\Scripts\python.exe" (
    call setup_jarvis.bat /quiet
    if errorlevel 1 goto failed
)
set "PYTHON=%~dp0.venv\Scripts\python.exe"

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed

"%PYTHON%" -m pip install pyinstaller
if errorlevel 1 goto failed

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $models=@(@{Name='vosk-model-small-ru-0.22'; Url='https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'}, @{Name='vosk-model-small-en-us-0.15'; Url='https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'}); foreach ($model in $models) { if (-not (Test-Path $model.Name)) { $zip=Join-Path $env:TEMP ($model.Name + '.zip'); Invoke-WebRequest -Uri $model.Url -OutFile $zip; Expand-Archive -Path $zip -DestinationPath . -Force; Remove-Item $zip } }"
if errorlevel 1 goto failed

"%PYTHON%" -m PyInstaller --clean --noconfirm --onefile --windowed --name Jarvis --distpath "release" --icon "jarvis.ico" --collect-all openai --collect-all anthropic --collect-all groq --collect-all pystray --collect-all vosk --collect-all torch --collect-all sounddevice --collect-all keyboard --collect-all PIL --collect-all pygame --collect-all edge_tts --add-data "v4_ru.pt;." --add-data "jarvis.ico;." --add-data "vosk-model-small-ru-0.22;vosk-model-small-ru-0.22" --add-data "vosk-model-small-en-us-0.15;vosk-model-small-en-us-0.15" my_ai.py
if errorlevel 1 goto failed

copy /y "release\Jarvis.exe" "Jarvis.exe" >nul
if errorlevel 1 (
    echo [WARN] Could not copy Jarvis.exe to the project folder. Close any running copy and copy it from dist manually.
) else (
    echo Jarvis.exe was also copied to the project folder.
)

echo.
echo Build complete: Jarvis.exe and release\Jarvis.exe
pause
exit /b 0

:failed
echo.
echo Build failed. See the error above.
pause
exit /b 1
