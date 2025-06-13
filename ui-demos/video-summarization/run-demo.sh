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

echo "Starting FastAPI app"
uvicorn api.app:app &
APP_PID=$!
sleep 10

echo "Running Video Summarizer"
python -m http.server 8005 &
VIDEO_PID=$! &
streamlit run summarizer/streamlit_merge.py --server.port 8501 &
streamlit run summarizer/streamlit_rag.py --server.port 8502

# terminate fastapi app after video summarization concludes
kill $APP_PID &
PID=$(lsof -ti tcp:8005)
if [ -n "$PID" ]; then
    echo"Killing video http server"
    kill -9 $PID
else
    echo "No process on port 8005"
fi

#pkill -f "python -m http.server 8005"
