#!/bin/bash

source activate-conda.sh
activate_conda
conda activate langchain_chapterization_env

export TOKENIZERS_PARALLELISM=true

INPUT_FILE=$1
ASR_8BIT_ENABLE_FLAG="--asr_load_in_8bit"
KMEANS_ENABLE_FLAG="--k_means_enabled"
ENABLE_FLAGS=""

if [ "$ASR_LOAD_IN_8BIT" == "1" ]
then
	ENABLE_FLAGS=$ASR_8BIT_ENABLE_FLAG
fi

if [ "$ENABLE_KMEANS" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS $KMEANS_ENABLE_FLAG"
fi

echo "Run chapterization"
python3 chapterization.py $INPUT_FILE --model_id $LLM_MODEL --device $INF_DEVICE --asr_batch_size $ASR_BATCH_SIZE --llm_batch_size $LLM_BATCH_SIZE --asr_model_id $ASR_MODEL $ENABLE_FLAGS
