@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m wt_overlay.sandbox %*
) else (
  py -3 -m wt_overlay.sandbox %*
)
if errorlevel 1 pause
