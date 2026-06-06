@echo off
REM Native Windows runner (cmd.exe). Requires conda on PATH.
REM Usage: run.bat --candidates .\candidates.jsonl --out .\submission.csv
setlocal enabledelayedexpansion
set ENV_NAME=canjob
set PY_VERSION=3.12
cd /d "%~dp0"

where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: conda not found on PATH.
  echo Install Miniconda/Anaconda: https://docs.conda.io/en/latest/miniconda.html
  exit /b 1
)

call conda env list | findstr /b /c:"%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
  echo Creating conda env: %ENV_NAME% ^(python=%PY_VERSION%^)
  call conda create -y -n %ENV_NAME% python=%PY_VERSION%
) else (
  echo Reusing existing conda env: %ENV_NAME%
)

call conda activate %ENV_NAME%
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python rank.py %*
echo Done.
endlocal
