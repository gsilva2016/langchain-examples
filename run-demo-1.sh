#!/bin/bash

source langchain_aicaps_demo_1_env/bin/activate
export TOKENIZERS_PARALLELISM=true
echo "Run #1"
python3 demo-1.py --model_id "gpt2" --device "CPU" --q "Which metrics are supported in the openvino model server? Give examples." --embeddings_api_endpoint "http://localhost:8000/v3/embeddings" --embeddings_model "BAAI/bge-small-en"

echo ""

echo "Run #2 - Changing models locally and from the OVMS embeddings server"
python3 demo-1.py --model_id "llmware/qwen2.5-1.5b-instruct-ov" --device "GPU" --q "Which metrics are supported in the openvino model server? Give examples." --embeddings_api_endpoint "http://localhost:8000/v3/embeddings" --embeddings_model "sentence-transformers/all-mpnet-base-v2"

echo ""

echo "Run #3 - Calling AWS Bedrock embeddings server"
python3 demo-1.py --model_id "llmware/qwen2.5-1.5b-instruct-ov" --device "GPU" --q "Which metrics are supported in the openvino model server? Give examples." --embeddings_model "amazon.titan-embed-text-v2:0"
