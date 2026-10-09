@echo off
setlocal
cd /d "%~dp0.."
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist ".venv\Scripts\python.exe" (
  echo [SceneArchitect] Missing .venv\Scripts\python.exe
  echo Please create the project virtual environment first.
  pause
  exit /b 1
)

echo [SceneArchitect] Starting http://127.0.0.1:7860
".venv\Scripts\python.exe" "BuShu\start.py"

if errorlevel 1 (
  echo.
  echo [SceneArchitect] Startup failed. Review the message above.
  pause
)
