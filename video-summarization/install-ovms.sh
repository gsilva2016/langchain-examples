#!/bin/bash

source .env

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN environment variable in .env file"
    exit 1
fi

echo "Install OpenVINO Model Server (OVMS) docker."
docker pull openvino/model_server:latest-gpu

# Install ovms enviornment for model conversion
source activate-conda.sh
activate_conda

conda create -n $OVMS_CONDA_ENV_NAME python=3.12 -y
conda activate $OVMS_CONDA_ENV_NAME

if [ $? -ne 0 ]; then
    echo "Conda environment $OVMS_CONDA_ENV_NAME activation has failed. Please check."
    exit 1
fi

conda install pip -y

# Install dependencies
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt -o ovms_requirements.txt
pip install -r ovms_requirements.txt

if [ "$1" == "--skip" ]; then
    echo "Skipping OpenVINO optimized model file creation"

else
    echo "Creating OpenVINO optimized model files for MiniCPM"
    huggingface-cli login --token $HUGGINGFACE_TOKEN

    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models

    # Note: For export_model.py, we normalize GPU.X to GPU since model export doesn't need specific GPU selection
    # Specific GPU device selection (GPU.0, GPU.1, etc.) happens at runtime when loading the models
    output=$(python export_model.py text_generation --source_model $VLM_MODEL --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device ${VLM_DEVICE%.*} --cache 2 --pipeline_type VLM --overwrite_models 2>&1 | tee /dev/tty)
    
    if echo "$output" | grep -q "Tokenizer won't be converted."; then
        echo ""
        echo "Error: Tokenizer was not converted successfully, OVMS export model has partially errored. Please check the logs."
        exit 1
    fi
    echo "VLM export completed successfully."
    echo ""

    # Update graph.pbtxt device parameter for VLM model
    VLM_GRAPH_PATH="models/${VLM_MODEL}/graph.pbtxt"
    if [ -f "$VLM_GRAPH_PATH" ]; then
        echo "Updating device parameter in $VLM_GRAPH_PATH to $VLM_DEVICE"
        sed -i "s/device: \".*\"/device: \"$VLM_DEVICE\"/" "$VLM_GRAPH_PATH"
    else
        echo "Warning: $VLM_GRAPH_PATH not found. Device parameter not updated."
    fi

    if [ -z "$SUMMARY_MERGER_LLM_DEVICE" ]; then
        echo "Please set the SUMMARY_MERGER_LLM_DEVICE environment variable to GPU, CPU or NPU in .env file"
        exit 1
    fi
    
    echo "Creating OpenVINO optimized model files for LLAMA Summary Merger on device: $SUMMARY_MERGER_LLM_DEVICE"
    # Normalize GPU.X to GPU for model export (specific device selection happens at runtime)
    export_device=${SUMMARY_MERGER_LLM_DEVICE%.*}
    
    if [ "$export_device" == "CPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --config_file_path models/config.json --model_repository_path models --target_device $export_device --weight-format fp16 --kv_cache_precision u8 --pipeline_type LM --overwrite_models

    elif [ "$export_device" == "GPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $export_device --cache 2 --pipeline_type LM --overwrite_models

    elif [ "$export_device" == "NPU" ]; then
        python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $export_device --max_prompt_len 1500 --pipeline_type LM --overwrite_models
    else
        echo "Invalid SUMMARY_MERGER_LLM_DEVICE value. Please set it to GPU (or GPU.0, GPU.1, etc.), CPU or NPU in .env file."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "LLAMA Summary Merger model export failed. Please check the logs."
        exit 1
    fi

    echo "LLAMA Summary Merger model export completed successfully."
    echo ""
fi
