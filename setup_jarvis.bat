@echo off
chcp 65001 >nul
title Установка JARVIS
color 0A
setlocal EnableExtensions
cd /d "%~dp0"
set "QUIET=%~1"
set "EXIT_CODE=0"
set "PYTHON_EXE="
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

echo.
echo ============================================================
echo       УСТАНОВЩИК ЛОКАЛЬНОГО AI-АССИСТЕНТА JARVIS
echo ============================================================
echo.
echo Установщик подготовит Python, библиотеки и офлайн-модели речи.
echo При первом запуске Jarvis предложит вставить Groq API-ключ.
echo Ключ не запрашивается и не сохраняется этим установщиком.
echo.
echo Понадобится интернет и около 2-4 ГБ свободного места.
echo Установка PyTorch может занять несколько минут.
echo.
if /i not "%QUIET%"=="/quiet" pause

if not exist "my_ai.py" (
    echo [ОШИБКА] Рядом с установщиком не найден my_ai.py.
    goto failed
)
if not exist "v4_ru.pt" (
    echo [ОШИБКА] Рядом с установщиком не найден v4_ru.pt.
    goto failed
)

echo.
echo [ШАГ 1/5] Проверяю Visual C++ Redistributable...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if not errorlevel 1 goto vc_ready

echo VC++ Runtime не найден. Скачиваю официальный установщик...
set "VC_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "VC_PATH=%TEMP%\vc_redist.x64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri '%VC_URL%' -OutFile '%VC_PATH%'"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось скачать VC++ Runtime.
    goto failed
)
start /wait "" "%VC_PATH%" /install /quiet /norestart
if "%ERRORLEVEL%"=="3010" goto vc_ready
if errorlevel 1 (
    echo [ОШИБКА] Установка VC++ Runtime завершилась с ошибкой.
    echo Запустите скачанный установщик вручную: %VC_PATH%
    goto failed
)
:vc_ready
echo [OK] Visual C++ Runtime готов.

echo.
echo [ШАГ 2/5] Проверяю Python 3.11...
call :find_python
if defined PYTHON_EXE goto python_ready

echo Python 3.11 не найден. Скачиваю официальный установщик...
set "PY_VER=3.11.9"
set "PY_FILE=python-%PY_VER%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/%PY_FILE%"
set "PY_PATH=%TEMP%\%PY_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri '%PY_URL%' -OutFile '%PY_PATH%'"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось скачать Python.
    goto failed
)
start /wait "" "%PY_PATH%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0 Include_pip=1
if errorlevel 1 (
    echo [ОШИБКА] Установка Python завершилась с ошибкой.
    goto failed
)
call :find_python
if not defined PYTHON_EXE (
    echo [ОШИБКА] Python установлен, но не найден. Перезапустите установщик.
    goto failed
)
:python_ready
echo [OK] Python: %PYTHON_EXE%

echo.
echo [ШАГ 3/5] Создаю изолированное окружение проекта...
if not exist "%VENV_PYTHON%" "%PYTHON_EXE%" -m venv ".venv"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось создать виртуальное окружение.
    goto failed
)
if not exist "%VENV_PYTHON%" (
    echo [ОШИБКА] Python окружения не найден.
    goto failed
)
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto failed

echo.
echo [ШАГ 4/5] Устанавливаю библиотеки проекта...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo [ШАГ 5/5] Загружаю локальные модели Vosk...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $models=@(@{Name='vosk-model-small-ru-0.22'; Url='https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'}, @{Name='vosk-model-small-en-us-0.15'; Url='https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'}); foreach ($model in $models) { if (-not (Test-Path $model.Name)) { $zip=Join-Path $env:TEMP ($model.Name + '.zip'); Invoke-WebRequest -UseBasicParsing -Uri $model.Url -OutFile $zip; Expand-Archive -Path $zip -DestinationPath . -Force; Remove-Item $zip -Force }; if (-not (Test-Path $model.Name)) { throw ('Model extraction failed: ' + $model.Name) } }"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось загрузить или распаковать модели Vosk.
    goto failed
)

echo Проверяю импорты установленных библиотек...
"%VENV_PYTHON%" -c "import anthropic, edge_tts, groq, keyboard, numpy, openai, PIL, psutil, pygame, pystray, requests, sounddevice, torch, vosk; print('OK: все библиотеки импортируются')"
if errorlevel 1 (
    echo [ОШИБКА] Одна или несколько библиотек не импортируются.
    goto failed
)
"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__)"
if errorlevel 1 goto failed
if not exist "vosk-model-small-ru-0.22" goto failed
if not exist "vosk-model-small-en-us-0.15" goto failed

echo.
echo ============================================================
echo УСТАНОВКА ЗАВЕРШЕНА
echo Запуск: start_jarvis.bat
echo При первом запуске вставьте Groq API-ключ кнопкой «Вставить».
echo ============================================================
echo.
if /i "%QUIET%"=="/quiet" goto success
choice /C YN /N /M "Запустить JARVIS сейчас? [Y/N] "
if errorlevel 2 goto success
call "start_jarvis.bat"
goto success

:find_python
set "PYTHON_EXE="
for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if defined PYTHON_EXE exit /b 0
for /f "delims=" %%P in ('"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
exit /b 0

:failed
echo.
echo Установка не завершена. См. сообщение об ошибке выше.
set "EXIT_CODE=1"
goto finish

:success
set "EXIT_CODE=0"
goto finish

:finish
if /i not "%QUIET%"=="/quiet" pause
exit /b %EXIT_CODE%
