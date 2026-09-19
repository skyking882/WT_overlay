@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m wt_overlay %*
) else (
    python -m wt_overlay %*
)
if errorlevel 1 (
    echo Install Python 3.11 or newer with Tcl/Tk enabled. See README.md.
    pause
    exit /b 1
)
