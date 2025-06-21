#!/bin/bash

source .env
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN environment variable in .env file"
    exit 1
fi

source activate-conda.sh

activate_conda
conda activate $CONDA_ENV_NAME
if [ $? -ne 0 ]; then
    echo "Conda environment activation has failed. Please check."
    exit
fi

huggingface-cli login --token $HUGGINGFACE_TOKEN

if [ "$1" == "--skip" ]; then
	echo "Skipping sample video download"
else
    # Download sample video
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
fi

PROJECT_ROOT_DIR=..

if [ -z "$SUMMARY_MERGER_LLM_DEVICE" ]; then
        echo "Please set the SUMMARY_MERGER_LLM_DEVICE environment variable to GPU, CPU or NPU in .env file"
        exit 1
fi

LLAMA_MODEL_DIR="llama-3.2-3b-merger-$SUMMARY_MERGER_LLM_DEVICE"
if [ ! -d "$LLAMA_MODEL_DIR" ]; then
    echo "LLAMA model directory $LLAMA_MODEL_DIR does not exist. Please run install.sh first to create the openVINO optimized model."
    exit 1
fi

# check if Milvus is running
if ! docker ps | grep -q "milvus"; then
    echo "Milvus is not running. Starting Milvus..."
    # check if Milvus script exists
    if [ ! -f "standalone_embed.sh" ]; then
        echo "Milvus start script not found. Please run install.sh first to install Milvus." 
        exit 1
    else
        bash standalone_embed.sh start
    fi
fi

if [ "$1" == "--run_rag" ]; then
    echo "Running RAG"
    
    if [ -z "$QUERY_TEXT" ]; then
    echo "Please set the QUERY_TEXT if you are running --run_rag."
    exit 1
    fi
    PYTHONPATH=$PROJECT_ROOT_DIR TOKENIZERS_PARALLELISM=true python src/rag.py --query_text "$QUERY_TEXT" --filter_expression "$FILTER_EXPR"
    
    echo "RAG completed"

else
    # Read OVMS endpoint and port from .env
    if [ -z "$OVMS_ENDPOINT" ]; then
        echo "OVMS_ENDPOINT is not set. Please set it in .env file."
        exit 1
    fi
    OVMS_PORT=$(echo "$OVMS_ENDPOINT" | sed -n 's/.*:\([0-9]\+\).*/\1/p')
    if [ -z "$OVMS_PORT" ]; then
        echo "Could not determine OVMS_PORT from OVMS_ENDPOINT ($OVMS_ENDPOINT)."
        exit 1
    fi
    OVMS_URL=$(echo "$OVMS_ENDPOINT" | sed -E 's#(https?://[^:/]+:[0-9]+).*#\1#')

    # Check if OVMS is already running
    if lsof -i:$OVMS_PORT | grep -q LISTEN; then
        echo "OVMS is already running on port $OVMS_PORT."
    else
        echo "Starting OVMS on port $OVMS_PORT."
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/ovms/lib
        export PATH=$PATH:${PWD}/ovms/bin
        export PYTHONPATH=$PYTHONPATH:${PWD}/ovms/lib/python
        ovms --rest_port $OVMS_PORT --config_path ./models/config.json &
        OVMS_PID=$!

        # Wait for OVMS to be ready
        echo "Waiting for OVMS to become available..."
        for i in {1..4}; do
            STATUS=$(curl -s $OVMS_URL/v1/config)
            if echo "$STATUS" | grep -q '"state": "AVAILABLE"'; then
                echo "OVMS is ready."
                break
            else
                sleep 8
            fi
            if [ $i -eq 4 ]; then
                echo "OVMS did not become ready in time. Please check the logs for errors."
		        kill $OVMS_PID
                exit 1
            fi
        done
        # OVMS installs are interfering with the openVINO packages installed via pip in conda
        # unsetting OVMS envs to prevent openVINO errors seen when Merger tries to start
        # a dockerized way of running ovms should be considered or running it in its own shell
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:${PWD}/ovms/lib}
        export PATH=${PATH%:${PWD}/ovms/bin}
        export PYTHONPATH=${PYTHONPATH%:${PWD}/ovms/lib/python}
    fi

    echo "Starting Merger Service"
    python $PROJECT_ROOT_DIR/services/langchain-merger-service/api/app.py --model_path $LLAMA_MODEL_DIR --device $SUMMARY_MERGER_LLM_DEVICE &
    MERGER_PID=$!
    
    MERGER_ENDPOINT=$(echo "$SUMMARY_MERGER_ENDPOINT" | cut -d'/' -f1-3)
    echo "Waiting for Merger Service to start..."
    while ! curl -s "$MERGER_ENDPOINT" > /dev/null; do
        sleep 5

        # if PID does not exist and failed for some reason, exit
        if ! ps -p $MERGER_PID > /dev/null; then
            echo "Merger Service failed to start. Please check the logs for errors."
            exit 1
        fi
    done
    echo "Merger Service is running at $MERGER_ENDPOINT"

    echo "Running Video Summarizer on video file: $INPUT_FILE"
    PYTHONPATH=$PROJECT_ROOT_DIR TOKENIZERS_PARALLELISM=true python src/main.py $INPUT_FILE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"

    echo "Video summarization completed"
fi

# terminate services
if [ -n "$MERGER_PID" ]; then
    echo "Terminating Merger Service PID: $MERGER_PID"
    kill -9 $MERGER_PID
    trap "kill -9 $MERGER_PID; exit" SIGINT SIGTERM
fi

if [ -n "$OVMS_PID" ]; then
    echo "Terminating OVMS PID: $OVMS_PID"
    kill -9 $OVMS_PID
    trap "kill -9 $OVMS_PID; exit" SIGINT SIGTERM
fi
