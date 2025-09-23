#!/bin/bash

# Clone vllm repo if not present
if [ -d "vllm" ]; then
    echo "WARNING: vllm dir already exists."
else
    git clone https://github.com/vllm-project/vllm.git --depth 1 --branch v0.10.1
fi

# Check if Docker image 'vllm-cpu' exists locally
if [ "$(docker images -q vllm-cpu:latest)" == "" ]; then
    echo "Docker image vllm-cpu not found locally. Building image..."
    cd vllm
    docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512BF16=false \
        --build-arg VLLM_CPU_AVX512VNNI=false \
        --build-arg VLLM_CPU_DISABLE_AVX512=true \
        --tag vllm-cpu \
        --target vllm-openai .
    cd ..
else
    echo "Docker image vllm-cpu found locally. Skipping build."
fi

# Load environment variables from .env
source .env

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Set HUGGINGFACE_TOKEN in .env"
    exit 1
fi

# Run the vLLM container
docker run -itd --rm --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=5g \
    --env "VLLM_LOGGING_LEVEL=INFO" \
    --env "VLLM_TARGET_DEVICE=cpu" \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGINGFACE_TOKEN" \
    -p 8012:8000 vllm-cpu \
    --model "$AGENT_MODEL" \
    --max_model_len 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes