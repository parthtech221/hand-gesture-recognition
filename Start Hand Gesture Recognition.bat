@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Python virtual environment not found.
    echo.
    echo Create it with:
    echo   python -m venv .venv
    echo.
    echo Then install the requirements with:
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "src\realtime_recognition.py" (
    echo.
    echo ERROR: src\realtime_recognition.py was not found.
    echo.
    pause
    exit /b 1
)

echo Starting Hand Gesture Recognition...
echo.
".venv\Scripts\python.exe" "src\realtime_recognition.py"

echo.
echo Hand Gesture Recognition has stopped.
pause
