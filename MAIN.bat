@echo off
:: monitor.bat — Restart ml-lstm-trader service (as Admin) and tail its log.
::
:: Usage:
::   deploy\monitor.bat          — restart service + tail stdout log
::   deploy\monitor.bat log      — tail log only (no restart)
::   deploy\monitor.bat status   — show service status and last 20 log lines
::
:: Requires: deploy\nssm.exe present (already in repo)

setlocal enabledelayedexpansion

:: Resolve repo root (parent of this deploy\ folder)
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

set "NSSM=%REPO_ROOT%\deploy\nssm.exe"
set "SERVICE=ml-lstm-trader"
set "LOG_DIR=%REPO_ROOT%\logs"
set "STDOUT_LOG=%LOG_DIR%\trader_stdout.log"
set "STDERR_LOG=%LOG_DIR%\trader_stderr.log"

:: Handle sub-commands
if /i "%~1"=="log"    goto :tail_log
if /i "%~1"=="status" goto :status

:: -----------------------------------------------------------------------
:: Default: restart service then tail log
:: -----------------------------------------------------------------------

:: Check if already running as Admin
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%REPO_ROOT%\" && deploy\monitor.bat' -Verb RunAs"
    exit /b
)

:: Ensure log dir exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo === ml-lstm-trader service restart ===
echo.

:: Check if service exists
"%NSSM%" status %SERVICE% >nul 2>&1
if errorlevel 1 (
    echo ERROR: Service "%SERVICE%" not found.
    echo Run deploy\nssm_install.bat as Administrator to install it first.
    pause
    exit /b 1
)

echo Restarting %SERVICE%...
"%NSSM%" restart %SERVICE%
if errorlevel 1 (
    echo WARNING: restart command returned an error. Trying stop + start...
    "%NSSM%" stop %SERVICE% >nul 2>&1
    timeout /t 3 /nobreak >nul
    "%NSSM%" start %SERVICE%
)

echo.
echo Service status:
"%NSSM%" status %SERVICE%

echo.
echo Tailing %STDOUT_LOG%
echo Press Ctrl+C to stop monitoring.
echo -----------------------------------------------------------------------
powershell -Command "Get-Content '%STDOUT_LOG%' -Wait -Tail 50"
goto :eof

:: -----------------------------------------------------------------------
:tail_log
echo Tailing %STDOUT_LOG%
echo Press Ctrl+C to stop.
echo -----------------------------------------------------------------------
powershell -Command "Get-Content '%STDOUT_LOG%' -Wait -Tail 80"
goto :eof

:: -----------------------------------------------------------------------
:status
echo.
echo === Service status ===
"%NSSM%" status %SERVICE%
echo.
echo === Last 20 lines of stdout ===
powershell -Command "if (Test-Path '%STDOUT_LOG%') { Get-Content '%STDOUT_LOG%' -Tail 20 } else { Write-Host 'Log not found: %STDOUT_LOG%' }"
echo.
echo === Last 5 lines of stderr ===
powershell -Command "if (Test-Path '%STDERR_LOG%') { Get-Content '%STDERR_LOG%' -Tail 5 } else { Write-Host 'No stderr log.' }"
goto :eof
