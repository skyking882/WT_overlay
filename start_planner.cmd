@echo off
setlocal
cd /d "%~dp0"
rem Opening-shot planner page, computed on this computer (python -m wt_overlay.planner_web).
rem missile_sim is a git submodule: fetch it if needed.
if exist ".gitmodules" (
    where git >nul 2>nul
    if not errorlevel 1 git submodule update --init
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m wt_overlay.planner_web %*
    goto done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m wt_overlay.planner_web %*
    goto done
)
python -m wt_overlay.planner_web %*
:done
if errorlevel 1 pause
