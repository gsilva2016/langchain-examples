#!/bin/bash

echo "[INFO] Starting llama-server..."

# Souce environment variables
source .env

# Extract hostname and port from LLAMA_CPP_ENDPOINT. First, strip protocol (http:// or https://)
HOST_PORT=$(echo "$LLAMA_CPP_ENDPOINT" | sed 's|https\?://||' | cut -d'/' -f1)

# Extract hostname
HOST=$(echo "$HOST_PORT" | cut -d':' -f1)

# Extract port
PORT=$(echo "$HOST_PORT" | cut -d':' -f2)
echo "[INFO] Using host: $HOST, port: $PORT"

# Change to installation directory and find extracted directory before we launch
cd "$LLAMA_CPP_INSTALL_DIR"
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "llama-cpp-*" | head -1 | sed 's|./||')
cd "$EXTRACTED_DIR"

# Kill any existing llama-server instances
if pgrep -f "./llama-server.*Qwen2.5-VL" > /dev/null; then
    echo "[INFO] Killing existing llama-server instances..."
    pkill -9 llama-server
    sleep 1
fi

# Run the server in the background and redirect output to a log file
setsid nohup ./llama-server -m ../models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
    --mmproj ../models/mmproj-F16.gguf -ngl 99 \
    --host "$HOST" --port "$PORT" --api-key "default-key" -c 8192 \
    > llama-server.log 2>&1 &
echo $! > /tmp/llama_server.pid

# Wait for server to be healthy
echo "[INFO] Waiting for server to become ready..."
MAX_WAIT=30
WAIT_TIME=0
WAIT_INTERVAL=5
HEALTH_URL="http://$HOST:$PORT/health"

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "$HEALTH_URL" | grep -q '"status":"ok"'; then
        echo "[INFO] Server is healthy and ready!"
        break
    fi
    sleep $WAIT_INTERVAL
    WAIT_TIME=$((WAIT_TIME + WAIT_INTERVAL))
    echo "[INFO] Waiting for server... ($WAIT_TIME/$MAX_WAIT seconds)"
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "[ERROR] Server failed to become ready within $MAX_WAIT seconds"
    echo "[ERROR] Check logs at: $(pwd)/llama-server.log"
    exit 1
fi

echo "[INFO] Server is running at $LLAMA_CPP_ENDPOINT"
