@echo off

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

REM Start FastAPI API in a new command prompt window
echo Starting FastAPI API...
start cmd /k "uvicorn api.main:app --reload --port 8000"

REM Give FastAPI a moment to start up
timeout /t 5 /nobreak >nul

REM Start Streamlit UI in another new command prompt window
echo Starting Streamlit UI...
start cmd /k "streamlit run ui/app.py"

REM Give Streamlit a moment to start up
timeout /t 5 /nobreak >nul

REM Open Streamlit UI in the default web browser
echo Opening Streamlit UI in browser...
start "" "http://localhost:8501"

echo All services started and UI opened.
echo Close the command prompt windows to stop the services.
