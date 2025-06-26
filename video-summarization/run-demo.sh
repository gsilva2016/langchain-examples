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
        echo "Starting Milvus..."
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
    # Start OVMS if not already running
    bash run-ovms.sh

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

OVMS_PID=$(pgrep -f "ovms")
if [ -n "$OVMS_PID" ]; then
    kill $OVMS_PID
    trap "kill $OVMS_PID; exit" SIGINT SIGTERM
fi
