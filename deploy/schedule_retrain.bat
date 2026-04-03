@echo off
:: Windows Task Scheduler registration for the autonomous retraining pipeline.
:: Run this script as Administrator.
::
:: Schedule:  Every Saturday at 2:00 AM local time.
:: Task name: ml-lstm-auto-retrain
:: To remove: schtasks /delete /tn ml-lstm-auto-retrain /f

:: --- Resolve repo root (parent of this deploy\ folder) ---
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

:: --- Resolve Python: prefer repo venv, fall back to PATH ---
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    for /f "delims=" %%i in ('where python 2^>nul') do set "PYTHON_EXE=%%i" & goto :py_found
    echo ERROR: Python not found. Activate your venv or add Python to PATH.
    pause
    exit /b 1
)
:py_found

:: --- Log directory ---
set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: --- Wrapper script (Task Scheduler needs a fixed working directory) ---
set "WRAPPER=%REPO_ROOT%\deploy\run_retrain.bat"
echo @echo off > "%WRAPPER%"
echo cd /d "%REPO_ROOT%" >> "%WRAPPER%"
echo "%PYTHON_EXE%" -m scripts.auto_retrain >> "%LOG_DIR%\auto_retrain.log" 2^>^&1 >> "%WRAPPER%"

echo.
echo Registering Task Scheduler job...
echo   Name:      ml-lstm-auto-retrain
echo   Python:    %PYTHON_EXE%
echo   Repo:      %REPO_ROOT%
echo   Schedule:  Every Saturday at 02:00 AM
echo   Log:       %LOG_DIR%\auto_retrain.log
echo.

:: Remove any previous instance (ignore error if it doesn't exist)
schtasks /delete /tn ml-lstm-auto-retrain /f > nul 2>&1

:: Register — single line, no ^ continuations
schtasks /create /tn "ml-lstm-auto-retrain" /tr "\"%WRAPPER%\"" /sc WEEKLY /d SAT /st 02:00 /f

if errorlevel 1 (
    echo.
    echo ERROR: schtasks failed. Make sure you are running as Administrator.
    pause
    exit /b 1
)

echo.
echo Task registered successfully.
echo.
echo Useful commands:
echo   schtasks /query  /tn ml-lstm-auto-retrain      -- show task details
echo   schtasks /run    /tn ml-lstm-auto-retrain      -- run immediately ^(test^)
echo   schtasks /end    /tn ml-lstm-auto-retrain      -- stop running instance
echo   schtasks /delete /tn ml-lstm-auto-retrain /f  -- remove the task
echo.
pause
