@echo off
REM Create a Python 3.11 virtual environment and install all project requirements.
REM Usage:
REM   setup_venv.bat          - auto: CUDA 12.8 wheels when nvidia-smi is found
REM   setup_venv.bat --cpu    - force CPU-only PyTorch
setlocal enabledelayedexpansion
pushd "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "HABIBI=third_party\Habibi-TTS"
set "CPU_ONLY="
if /I "%~1"=="--cpu" set "CPU_ONLY=1"
if /I "%~1"=="-cpu" set "CPU_ONLY=1"

if exist "%PY%" (
    for /f "delims=" %%v in ('"%PY%" -c "import sys;print(str(sys.version_info[0])+chr(46)+str(sys.version_info[1]))"') do set "VENV_VER=%%v"
    if not "!VENV_VER!"=="3.11" (
        echo Existing .venv uses Python !VENV_VER!. Recreating with Python 3.11...
        rmdir /s /q "%VENV%"
    )
)

if not exist "%PY%" (
    where uv >nul 2>nul
    if not errorlevel 1 (
        echo Creating Python 3.11 environment with uv...
        uv venv --python 3.11 --seed "%VENV%"
        if errorlevel 1 goto :error
    ) else (
        py -3.11 -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            echo Creating Python 3.11 environment with py launcher...
            py -3.11 -m venv "%VENV%"
            if errorlevel 1 goto :error
        ) else (
            python3.11 -c "import sys" >nul 2>nul
            if not errorlevel 1 (
                echo Creating Python 3.11 environment with python3.11...
                python3.11 -m venv "%VENV%"
                if errorlevel 1 goto :error
            ) else (
                echo Python 3.11 was not found. Install Python 3.11 or uv and try again.
                goto :error
            )
        )
    )
)

echo.
echo Using interpreter: %PY%
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

if not defined CPU_ONLY (
    where nvidia-smi >nul 2>nul
    if errorlevel 1 set "CPU_ONLY=1"
)

if defined CPU_ONLY (
    echo.
    echo Installing CPU PyTorch 2.8.0...
    "%PY%" -m pip install torch==2.8.0 torchaudio==2.8.0
    if errorlevel 1 goto :error
) else (
    echo.
    echo Installing CUDA 12.8 PyTorch 2.8.0...
    "%PY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
    if errorlevel 1 goto :error
)

echo.
echo Installing requirements...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Installing local Habibi-TTS package...
"%PY%" -m pip install --no-deps -e "%HABIBI%"
if errorlevel 1 goto :error

"%PY%" -m pip check

where ffmpeg >nul 2>nul
if errorlevel 1 echo WARNING: ffmpeg was not found on PATH. Install it before using Whisper or audio decoding.

echo.
echo Setup complete.
echo Activate with: .venv\Scripts\activate.bat
echo Download models/fonts with: python scripts\download_models.py all
echo Run prototypes from prototypes\<name>\README.md
if defined CPU_ONLY (echo PyTorch: CPU build) else (echo PyTorch: CUDA 12.8 build)
popd
pause
exit /b 0

:error
echo.
echo Setup failed. See the error above.
popd
pause
exit /b 1
