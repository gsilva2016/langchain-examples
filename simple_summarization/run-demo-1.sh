#!/bin/bash

source langchain_aicaps_demo_vs1_env/bin/activate
export TOKENIZERS_PARALLELISM=true

INPUT_FILE=Unzc731iCUY_audio.mp3

#MODEL_ID=rajatkrishna/Meta-Llama-3-8B-OpenVINO-INT4
#MODEL_ID=llmware/qwen2-1.5b-instruct-ov
MODEL_ID="llmware/llama-3.2-3b-instruct-ov"
#MODEL_ID=llmware/dolphin-2.9.3-mistral-7b-32k-ov

echo "Run Video Summarization"
python3 demo-summarization-1.py --model_id $MODEL_ID --device "GPU" $INPUT_FILE --asr_batch_size 4 --asr_load_in_8bit
