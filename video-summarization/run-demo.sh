#!/bin/bash

source .env
source activate-conda.sh
activate_conda
conda activate ovlangvidsumm

if [ "$1" == "--skip" ]; then
	echo "Skipping sample video download"
else
    # Download sample video
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
fi

INPUT_FILE="one-by-one-person-detection.mp4"
RESOLUTION_X=480
RESOLUTION_Y=270
PROMPT='As an expert investigator, please analyze this video. Summarize the video, highlighting any shoplifting or suspicious activity. The output must contain the following 3 sections: Overall Summary, Activity Observed, Potential Suspicious Activity. It should be formatted similar to the following example:

**Overall Summary**
Here is a detailed description of the video.

**Activity Observed**
1) Here is a bullet point list of the activities observed. If nothing is observed, say so, and the list should have no more than 10 items.

**Potential Suspicious Activity**
1) Here is a bullet point list of suspicious behavior (if any) to highlight.
'
export HF_ACCESS_TOKEN=
QUERY_TEXT=
FILTER_EXPR=

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

if [ -z "$HF_ACCESS_TOKEN" ]; then
    echo "Please set the HF_ACCESS_TOKEN environment variable in run_demo.sh"
    exit 1
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
        echo "OVMS_ENDPOINT is not set. Please set it in your environment or .env file."
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
        export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH:${PWD}/ovms/lib
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
    fi

    echo "Starting Merger Service"
    python $PROJECT_ROOT_DIR/services/langchain-merger-service/api/app.py &
    MERGER_PID=$!
    
    MERGER_ENDPOINT=$(echo "$SUMMARY_MERGER_ENDPOINT" | cut -d'/' -f1-3)
    echo "Waiting for Merger Service to start..."
    while ! curl -s "$MERGER_ENDPOINT" > /dev/null; do
        sleep 5
    done
    echo "Merger Service is running at $MERGER_ENDPOINT"

    echo "Running Video Summarizer"
    PYTHONPATH=$PROJECT_ROOT_DIR TOKENIZERS_PARALLELISM=true python src/main.py $INPUT_FILE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"

    echo "Video summarization completed"
fi

# terminate services
if [ -n "$MERGER_PID" ]; then
    kill $MERGER_PID
    trap "kill $MERGER_PID; exit" SIGINT SIGTERM
fi

if [ -n "$OVMS_PID" ]; then
    kill $OVMS_PID
    trap "kill $OVMS_PID; exit" SIGINT SIGTERM
fi
