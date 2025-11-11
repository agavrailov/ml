@echo off

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

REM Run the data updater script
echo Running data updater...
python src/data_updater.py

REM Run the data processing script
echo Running data processing...
python src/data_processing.py

echo Data update and processing complete.
pause