@echo off
REM Start the Plan Prediction API

echo ========================================
echo Starting Plan Prediction API
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting API server...
echo The API will be available at: http://localhost:8000
echo Interactive docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd api
python -m uvicorn main:app --reload --host 0.0.0.0
