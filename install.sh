#!/bin/bash

# OpenVINO, CSP, hybrid architecture example
python3 -m venv langchain_aicaps_demo_1_env
source langchain_aicaps_demo_1_env/bin/activate
python -m pip install --upgrade pip
pip install wheel setuptools langchain-openai langchain_community langchain_aws faiss-cpu unstructured
pip install --upgrade-strategy eager "optimum[openvino,nncf]" langchain-huggingface

curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html --create-dirs -o ./docs/ovms_what_is_openvino_model_server.html
curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_metrics.html -o ./docs/ovms_docs_metrics.html
curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_streaming_endpoints.html -o ./docs/ovms_docs_streaming_endpoints.html
curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_target_devices.html -o ./docs/ovms_docs_target_devices.html

# IPEX
#python3 -m venv langchain_aicaps_demo_2_env
#source langchain_aicaps_demo_2_env/bin/activate
#python -m pip install --upgrade pip
#pip install wheel setuptools openlm langchain-openai
#pip install --upgrade-strategy eager "optimum[ipex]" langchain-huggingface
