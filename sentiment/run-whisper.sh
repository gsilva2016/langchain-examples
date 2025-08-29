#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"

source .env
source activate-conda.sh
activate_conda
conda activate $CONDA_WHISPER_ENV_NAME

source $OPENVINO_DIR/setupvars.sh
$WHISPER_DIR/build/bin/whisper-server -m $WHISPER_DIR/models/ggml-base.en.bin --port 5910 -ml 1 -oved $WHISPER_DEVICE --print-realtime
