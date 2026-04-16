@echo off
:: NSSM service for the multi-symbol portfolio launcher.
:: Run as Administrator.

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    for /f "delims=" %%i in ('where python 2^>nul') do set "PYTHON_EXE=%%i" & goto :py_found
    echo ERROR: Python not found.
    pause & exit /b 1
)
:py_found

set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "WRAPPER=%REPO_ROOT%\deploy\run_portfolio.bat"
echo @echo off > "%WRAPPER%"
echo cd /d "%REPO_ROOT%" >> "%WRAPPER%"
echo "%PYTHON_EXE%" -m scripts.launch_portfolio >> "%LOG_DIR%\portfolio.log" 2^>^&1 >> "%WRAPPER%"

schtasks /delete /tn ml-lstm-portfolio /f > nul 2>&1
nssm stop  ml-lstm-portfolio > nul 2>&1
nssm remove ml-lstm-portfolio confirm > nul 2>&1

nssm install ml-lstm-portfolio "%WRAPPER%"
nssm set ml-lstm-portfolio AppDirectory "%REPO_ROOT%"
nssm set ml-lstm-portfolio AppRestartDelay 30000
nssm set ml-lstm-portfolio AppExit Default Restart
nssm set ml-lstm-portfolio Start SERVICE_AUTO_START
nssm set ml-lstm-portfolio AppStdout "%LOG_DIR%\portfolio_stdout.log"
nssm set ml-lstm-portfolio AppStderr "%LOG_DIR%\portfolio_stderr.log"
nssm set ml-lstm-portfolio AppRotateFiles 1
nssm set ml-lstm-portfolio AppRotateSeconds 86400

nssm start ml-lstm-portfolio
if errorlevel 1 (
    echo ERROR: service failed to start. Is IB Gateway running?
    pause & exit /b 1
)

echo Service ml-lstm-portfolio started.
pause
