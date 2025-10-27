#!/bin/bash
source .env

# Read OVMS endpoint and port from .env
if [ -z "$OVMS_ENDPOINT" ]; then
    echo "OVMS_ENDPOINT is not set. Please set it in the .env file."
    exit 1
fi
OVMS_REST_PORT=$(echo "$OVMS_ENDPOINT" | sed -n 's/.*:\([0-9]\+\).*/\1/p')

if [ -z "$OVMS_REST_PORT" ]; then
    echo "Could not determine OVMS_REST_PORT from OVMS_ENDPOINT ($OVMS_ENDPOINT)."
    exit 1
fi
OVMS_URL=$(echo "$OVMS_ENDPOINT" | sed -E 's#(https?://[^:/]+:[0-9]+).*#\1#')

# Kill any hanging ovms containers
if docker ps --filter "name=$OVMS_CONTAINER_NAME" --format '{{.Names}}' | grep -q $OVMS_CONTAINER_NAME; then
    echo "Container is already running. Stopping container and rerunning..."
    docker stop $OVMS_CONTAINER_NAME
fi

# Check if NPU is selected in any .env parameter
NPU_DEVICE=""
if grep -q "NPU" .env; then
    NPU_DEVICE="--device /dev/accel/accel0"
fi

# Check if GPU is selected in any .env parameter
GPU_DEVICE=""
GROUP_ADD=""
if grep -q "GPU" .env; then
    GPU_DEVICE="--device /dev/dri"
    GROUP_ADD="--group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)"
fi

# Start Docker
echo "Starting container: $OVMS_CONTAINER_NAME on port: $OVMS_REST_PORT."
docker run -d --rm -v ${PWD}/models:/models -p $OVMS_GRPC_PORT:$OVMS_GRPC_PORT -p $OVMS_REST_PORT:$OVMS_REST_PORT --name $OVMS_CONTAINER_NAME $GPU_DEVICE $NPU_DEVICE $GROUP_ADD openvino/model_server:latest-gpu --config_path /models/config.json --port $OVMS_GRPC_PORT --rest_port $OVMS_REST_PORT

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
	docker stop $OVMS_CONTAINER_NAME	
        exit 1
    fi
done
