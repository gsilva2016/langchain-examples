#!/bin/bash

source activate-conda.sh
activate_conda
conda activate langchain_qna_env

export TOKENIZERS_PARALLELISM=true

DEMO_MODE_FLAG="--demo_mode"
RAG_MODE_FLAG="--rag"
ENABLE_FLAGS=""

if [ "$DEMO_MODE" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS $DEMO_MODE_FLAG"
fi

if [ "$RAG_ENABLED" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS $RAG_MODE_FLAG"
fi

echo "Run text qna"
python3 text_qna_rag.py --model_id $LLM_MODEL --device $INF_DEVICE --llm_batch_size $LLM_BATCH_SIZE $ENABLE_FLAGS
