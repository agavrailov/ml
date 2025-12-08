@echo off

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

start streamlit run src/app.py

