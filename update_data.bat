@echo off

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

REM Run the daily data pipeline agent (ingestion, cleaning, gaps, curation, features)
echo Running daily data pipeline agent...
python -m src.daily_data_agent

echo Data update and processing complete.
