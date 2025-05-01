#!/bin/bash

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
DEVICE="GPU"
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

if [ "$1" == "--run_summarizer" ] || [ "$2" == "--run_summarizer" ]; then
    echo "Starting Merger Service"
    python services/langchain-merger-service/api/app.py &
    MERGER_PID=$!
    sleep 10

    echo "Starting Milvus Service"
    python services/langchain-milvus-service/api/app.py &
    MILVUS_PID=$!
    sleep 10

    echo "Running Video Summarizer"
    PYTHONPATH=. python src/main.py $INPUT_FILE MiniCPM_INT8/ -d $DEVICE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"

    echo "Video summarization completed"    

if [ "$1" == "--run_summarizer" ] || [ "$2" == "--run_summarizer" ]; then  
    echo "Starting Milvus Service"
    python services/langchain-milvus-service/api/app.py &
    MILVUS_PID=$!
    sleep 10

    echo "Running RAG"
    PYTHONPATH=. python src/rag.py --query_text "$QUERY_TEXT"
    
    echo "RAG completed"

# terminate FastAPI apps
if [ -n "$MERGER_PID" ]; then
    kill $MERGER_PID
    trap "kill $MERGER_PID; exit" SIGINT SIGTERM
fi

if [ -n "$MILVUS_PID" ]; then
    kill $MILVUS_PID
    trap "kill $MILVUS_PID; exit" SIGINT SIGTERM
fi
