#!/bin/bash

source .env

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN environment variable in .env file"
    exit 1
fi

echo "Install OpenVINO Model Server (OVMS) on baremetal"

source activate-conda.sh
activate_conda

conda create -n ovms_env python=3.12 -y
conda activate ovms_env

if [ $? -ne 0 ]; then
    echo "Conda environment ovms_env activation has failed. Please check."
    exit 1
fi

conda install pip -y

# Install dependencies
sudo apt update -y && sudo apt install -y libxml2 curl
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/requirements.txt -o ovms_requirements.txt
pip install "Jinja2==3.1.6" "MarkupSafe==3.0.2" "huggingface_hub"
pip install -r ovms_requirements.txt

# Download OVMS
export PATH=$PATH:${PWD}/ovms/bin
if command -v ovms &> /dev/null; then
    echo "OpenVINO Model Server (OVMS) is already installed."
else
    echo "Downloading OpenVINO Model Server (OVMS)..."  
    wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu24_python_on.tar.gz
    tar -xzvf ovms_ubuntu24_python_on.tar.gz
fi

if [ "$1" == "--skip" ]; then
    echo "Skipping OpenVINO optimized model file creation"

else
    echo "Creating OpenVINO optimized model files for MiniCPM"
    huggingface-cli login --token $HUGGINGFACE_TOKEN

    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models

    output=$(python export_model.py text_generation --source_model $VLM_MODEL --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device $VLM_DEVICE --cache 2 --pipeline_type VLM --overwrite_models 2>&1 | tee /dev/tty)
    
    if echo "$output" | grep -q "Tokenizer won't be converted."; then
        echo ""
        echo "Error: Tokenizer was not converted successfully, OVMS export model has partially errored. Please check the logs."
        exit 1
    fi
    echo "MiniCPM model export completed successfully."
    echo ""

    if [ -z "$SUMMARY_MERGER_LLM_DEVICE" ]; then
        echo "Please set the SUMMARY_MERGER_LLM_DEVICE environment variable to GPU, CPU or NPU in .env file"
        exit 1
    fi
    
    echo "Creating OpenVINO optimized model files for LLAMA Summary Merger on device: $SUMMARY_MERGER_LLM_DEVICE"
    if [ "$SUMMARY_MERGER_LLM_DEVICE" == "CPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --config_file_path models/config.json --model_repository_path models --target_device $SUMMARY_MERGER_LLM_DEVICE --weight-format fp16 --kv_cache_precision u8 --pipeline_type LM --overwrite_models

    elif [ "$SUMMARY_MERGER_LLM_DEVICE" == "GPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $SUMMARY_MERGER_LLM_DEVICE --cache 2 --pipeline_type LM --overwrite_models

    elif [ "$SUMMARY_MERGER_LLM_DEVICE" == "NPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $SUMMARY_MERGER_LLM_DEVICE --max_prompt_len 1500 --pipeline_type LM --overwrite_models
    else
        echo "Invalid SUMMARY_MERGER_LLM_DEVICE value. Please set it to GPU, CPU or NPU in .env file."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "LLAMA Summary Merger model export failed. Please check the logs."
        exit 1
    fi

    echo "LLAMA Summary Merger model export completed successfully."
    echo ""
fi