#!/bin/bash

source .env
source activate-conda.sh
activate_conda
conda activate $CONDA_ENV_NAME

export TOKENIZERS_PARALLELISM=true

ENABLE_FLAGS=""
if [ "$DEMO_MODE" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS --demo_mode"
fi

if [ "$ASR_DEVICE" == "" ]
then
	echo "Set DEVICE in .env file"
	exit 1
fi


# Use for remote
#SENTIMENT_MODEL="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
# Use for local
#SENTIMENT_MODEL="llmware/qwen2-1.5b-instruct-ov"

echo "Run sentiment analysis (text)"
python3 sentiment.py $1 --device $ASR_DEVICE --sentiment_model_id $OVMS_MODEL --sentiment_endpoint $OVMS_ENDPOINT $ENABLE_FLAGS
