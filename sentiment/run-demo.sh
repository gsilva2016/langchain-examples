#!/bin/bash

source activate-conda.sh
activate_conda
conda activate langchain_sentiment_analysis_env

export TOKENIZERS_PARALLELISM=true

# Use for remote
#SENTIMENT_MODEL="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
# Use for local
SENTIMENT_MODEL="llmware/qwen2-1.5b-instruct-ov"

echo "Run sentiment analysis (text)"
python3 sentiment.py $1 --device $INF_DEVICE --sentiment_model_id $SENTIMENT_MODEL
