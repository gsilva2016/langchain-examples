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
export HF_ACCESS_TOKEN=<your_huggingface_access_token>
QUERY_TEXT="<your_query_text>"
PROJECT_ROOT_DIR=..

if [ "$1" == "--run_summarizer" ] || [ "$2" == "--run_summarizer" ]; then
    echo "Starting Merger Service"
    python $PROJECT_ROOT_DIR/services/langchain-merger-service/api/app.py &
    MERGER_PID=$!
    sleep 10

    echo "Running Video Summarizer"
    PYTHONPATH=$PROJECT_ROOT_DIR python src/main.py $INPUT_FILE MiniCPM_INT8/ -d $DEVICE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"

    echo "Video summarization completed"
fi 

if [ "$1" == "--run_rag" ] || [ "$2" == "--run_rag" ]; then  
    echo "Running RAG"
    PYTHONPATH=$PROJECT_ROOT_DIR python src/rag.py --query_text "$QUERY_TEXT"
    
    echo "RAG completed"
fi 

# terminate FastAPI apps
if [ -n "$MERGER_PID" ]; then
    kill $MERGER_PID
    trap "kill $MERGER_PID; exit" SIGINT SIGTERM
fi
