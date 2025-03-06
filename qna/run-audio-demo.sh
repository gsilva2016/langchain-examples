#!/bin/bash

source activate-conda.sh
activate_conda
conda activate langchain_qna_env

export TOKENIZERS_PARALLELISM=true

INPUT_FILE=$1
ASR_8BIT_ENABLE_FLAG="--asr_load_in_8bit"
DEMO_MODE_FLAG="--demo_mode"
RAG_MODE_FLAG="--rag"
ENABLE_FLAGS=""

if [ "$ASR_LOAD_IN_8BIT" == "1" ]
then
	ENABLE_FLAGS=$ASR_8BIT_ENABLE_FLAG
fi

if [ "$DEMO_MODE" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS $DEMO_MODE_FLAG"
fi

if [ "$RAG_ENABLED" == "1" ]
then
	ENABLE_FLAGS="$ENABLE_FLAGS $RAG_MODE_FLAG"
fi

KOKORO_LAUNCHED=$(docker ps -f name=kokoro | wc -l)
if [ "$TTS_MODEL" == "kokoro" ] && [ "$KOKORO_LAUNCHED" == "1" ]
then
	docker run --rm -itd --name "kokoro" -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.1.0post1 # CPU 
fi

if [ "$TTS_MODEL" == "" ]
then
	#TTS_MODEL="OuteAI/OuteTTS-0.1-350M"
	TTS_MODEL="kokoro"
fi

echo "Run audio qna"
python3 audio_qna_rag.py $INPUT_FILE --model_id $LLM_MODEL --device $INF_DEVICE --asr_batch_size $ASR_BATCH_SIZE --llm_batch_size $LLM_BATCH_SIZE --asr_model_id $ASR_MODEL --tts_model_id $TTS_MODEL $ENABLE_FLAGS
