@echo off
setlocal

IF not EXIST "C:\Users\%username%\AppData\Local\miniforge3\Scripts\" (
	echo "Conda not found. Please install from https://conda-forge.org/download/ for local user as recommended."
	goto :eof
)

for /f "delims=" %%i in (.env) do (
    set "%%i"
)

set WHISPER_DIR=whisper.cpp
call conda activate %CONDA_ENV_NAME%

call %OPENVINO_DIR%/setupvars.bat
REM "%WHISPER_DIR%\build\bin\Release\whisper-server.exe" -m %WHISPER_DIR%/models/ggml-base.en.bin -oved %WHISPER_DEVICE% --port 5910 
start "Whisper-Server" "whisper.cpp\build\bin\Release\whisper-server.exe" -m %WHISPER_DIR%/models/ggml-base.en.bin -oved %WHISPER_DEVICE% --port 5910 
