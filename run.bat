@echo off

REM Navigate to the project root directory
cd /d C:\Users\Anton\SRC\my\ml_lstm

REM Activate the Python virtual environment
REM echo Activating virtual environment...
REM call .\venv\Scripts\activate

REM Start UI
echo Starting UI...
start streamlit run src/ui/app.py

REM Start live trading with IBKR
echo Starting live trading with IBKR
start python -m src.ibkr_live_session --symbol NVDA --frequency 60min --snapshot-every-n-bars 1 --client-id 2 >> logs\live_trader.log 2>&1

