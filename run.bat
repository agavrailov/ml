@echo off

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

REM Start UI
echo Starting UI...
start streamlit run src/app.py

REM Start live trading with IBKR
echo Starting live trading with IBKR
start python -m src.ibkr_live_session --symbol NVDA --frequency 60min --snapshot-every-n-bars 1 --client-id 2

