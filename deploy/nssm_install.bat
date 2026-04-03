@echo off
:: NSSM Windows Service Installer for ml-lstm-trader
::
:: Prerequisites:
::   1. Download NSSM from https://nssm.cc/download and put nssm.exe on PATH
::      (or place nssm.exe in this deploy\ folder)
::   2. Run this script as Administrator
::
:: What it does:
::   - Registers ml-lstm-trader as a Windows service that starts automatically
::   - Restarts the daemon within 30 seconds if it crashes
::   - Sets the working directory to the repo root

setlocal enabledelayedexpansion

:: Detect repo root (parent of this deploy\ folder)
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

:: Detect Python in the project venv, then fall back to system Python
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
)
if not exist "%PYTHON_EXE%" (
    where python > nul 2>&1 && (for /f "delims=" %%i in ('where python') do set "PYTHON_EXE=%%i" & goto :found_python)
    echo ERROR: Python not found. Activate your venv or add Python to PATH.
    exit /b 1
)
:found_python

:: Check for nssm on PATH or in this folder
where nssm > nul 2>&1
if errorlevel 1 (
    if exist "%~dp0nssm.exe" (
        set "NSSM=%~dp0nssm.exe"
    ) else (
        echo ERROR: nssm.exe not found. Download from https://nssm.cc/download
        echo Place nssm.exe on PATH or in the deploy\ folder next to this script.
        exit /b 1
    )
) else (
    set "NSSM=nssm"
)

set "SERVICE_NAME=ml-lstm-trader"
set "SERVICE_ARGS=-m src.ibkr_live_session --symbol NVDA --frequency 60min --client-id 2"

echo.
echo Installing Windows service: %SERVICE_NAME%
echo   Python:      %PYTHON_EXE%
echo   Arguments:   %SERVICE_ARGS%
echo   Working dir: %REPO_ROOT%
echo.

:: Remove existing service if present
%NSSM% status %SERVICE_NAME% > nul 2>&1
if not errorlevel 1 (
    echo Removing existing service...
    %NSSM% stop %SERVICE_NAME% > nul 2>&1
    %NSSM% remove %SERVICE_NAME% confirm > nul 2>&1
)

:: Install
%NSSM% install %SERVICE_NAME% "%PYTHON_EXE%" "%SERVICE_ARGS%"
if errorlevel 1 (
    echo ERROR: nssm install failed.
    exit /b 1
)

:: Configure
%NSSM% set %SERVICE_NAME% AppDirectory "%REPO_ROOT%"
%NSSM% set %SERVICE_NAME% AppRestartDelay 30000
%NSSM% set %SERVICE_NAME% AppExit Default Restart
%NSSM% set %SERVICE_NAME% Start SERVICE_AUTO_START
%NSSM% set %SERVICE_NAME% DisplayName "ML LSTM Trader Daemon"
%NSSM% set %SERVICE_NAME% Description "LSTM-based NVDA live trading daemon (poll-loop mode)"

:: Redirect stdout/stderr to log files
set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
%NSSM% set %SERVICE_NAME% AppStdout "%LOG_DIR%\trader_stdout.log"
%NSSM% set %SERVICE_NAME% AppStderr "%LOG_DIR%\trader_stderr.log"
%NSSM% set %SERVICE_NAME% AppRotateFiles 1
%NSSM% set %SERVICE_NAME% AppRotateSeconds 86400

echo.
echo Service configured. Starting...
%NSSM% start %SERVICE_NAME%
if errorlevel 1 (
    echo WARNING: Service installed but failed to start. Check that IB Gateway is running.
    echo To start manually: nssm start %SERVICE_NAME%
) else (
    echo Service started successfully.
)

echo.
echo Useful commands:
echo   nssm status %SERVICE_NAME%     -- check status
echo   nssm stop   %SERVICE_NAME%     -- stop daemon
echo   nssm start  %SERVICE_NAME%     -- start daemon
echo   nssm remove %SERVICE_NAME%     -- uninstall service
echo   nssm edit   %SERVICE_NAME%     -- open GUI editor
echo.

endlocal
