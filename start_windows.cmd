@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto dependencies
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 goto failed

:dependencies
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3, 11)" >nul 2>nul
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import PySide6.QtWidgets" >nul 2>nul
if not errorlevel 1 goto launch
echo Installing the transparent HUD interface. First run requires internet access.
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:launch
for %%A in (%*) do (
    if /i "%%~A"=="--headless" goto console
    if /i "%%~A"=="--help" goto console
    if /i "%%~A"=="-h" goto console
)
if not exist ".venv\Scripts\pythonw.exe" goto failed
start "" ".venv\Scripts\pythonw.exe" -m wt_overlay.windowed %*
if errorlevel 1 goto failed
exit /b 0

:console
".venv\Scripts\python.exe" -m wt_overlay %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo WT Energy could not start. Python 3.11 or newer is required. See the error above and README.md.
pause
exit /b 1
