# JARVIS Windows build

## Build a standalone executable

Requirements: 64-bit Windows and an internet connection during setup/build. Keep the project files together. Run `setup_jarvis.bat` to install Python 3.11 if needed, create a project-local virtual environment, install runtime packages, and download the Russian and English Vosk models. Then run `build.bat` to install PyInstaller into that environment and create `Jarvis.exe` in the project folder and `release\Jarvis.exe`.

The executable includes Python dependencies and speech models. On first launch, choose an AI provider and model, paste its API key, test the connection, and save. API keys are stored as plain text in `%APPDATA%\Jarvis\config.json`; do not share that file. Ollama can run locally; Vosk speech recognition also runs locally.

JARVIS allows only one running instance. Its tray icon has a **Завершить JARVIS** command to stop it at any time.

## Run from source

After setup completes, run `start_jarvis.bat`. At first launch, choose a provider and model, paste the provider key, test the connection, and save. Keys are stored locally in `%APPDATA%\Jarvis\config.json`.
