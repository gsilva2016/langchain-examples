#!/bin/bash

sudo apt install python3-venv git -y

python3 -m venv langchain_aicaps_demo_vs1_env
source langchain_aicaps_demo_vs1_env/bin/activate
python -m pip install --upgrade pip
pip install wheel setuptools langchain-openai langchain_community
pip install --upgrade-strategy eager "optimum[openvino,nncf]" langchain-huggingface
git clone https://github.com/gsilva2016/langchain.git
pushd langchain; git checkout openvino_asr_loader; popd
pip install -e langchain/libs/community
