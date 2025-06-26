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
    bash run-ovms.sh

    if [ $? -ne 0 ]; then
        echo "OVMS setup failed. Please check the logs."
        exit 1
    fi
    
    echo "Running Video Summarizer on video file: $INPUT_FILE"
    PYTHONPATH=$PROJECT_ROOT_DIR TOKENIZERS_PARALLELISM=true python src/main.py $INPUT_FILE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"
    echo ""

    echo "Video summarization completed"
    echo ""
fi

# terminate services
OVMS_PID=$(pgrep -f "ovms")
if [ -n "$OVMS_PID" ]; then
    echo "Terminating OVMS PID: $OVMS_PID"
    kill -9 $OVMS_PID
    trap "kill -9 $OVMS_PID; exit" SIGINT SIGTERM
fi