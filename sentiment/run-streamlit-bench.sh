#!/bin/bash

source .env
source activate-conda.sh
activate_conda
conda activate $CONDA_ENV_NAME

#huggingface-cli login --token $HF_ACCESS_TOKEN
export HF_ACCESS_TOKEN=$HF_ACCESS_TOKEN
streamlit run streamlit-demo-bench.py --server.port=8080
