@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo No Python 3.12 environment found. Run setup.ps1 first. 1>&2
  exit /b 1
)

"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 (
  echo .venv is not Python 3.12. Delete it and re-run setup.ps1. 1>&2
  exit /b 1
)

"%CD%\.venv\Scripts\hf.exe" auth whoami >nul 2>&1
if errorlevel 1 echo Not logged in to Hugging Face; run .venv\Scripts\hf.exe auth login if model weights still need downloading. 1>&2

:run
"%PYTHON%" run.py %*
if %errorlevel% equ 75 (
  echo CUDA context lost; restarting the service in 5 seconds... 1>&2
  timeout /t 5 /nobreak >nul
  goto run
)
exit /b %errorlevel%
