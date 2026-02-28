REM Start live trading with IBKR
echo Starting live trading with IBKR
start python -m src.ibkr_live_session --symbol NVDA --frequency 60min --snapshot-every-n-bars 1 --client-id 2

