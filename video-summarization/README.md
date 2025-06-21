# Summarize Videos Using OpenVINO Model Server, Langchain, and MiniCPM-V-2_6

## Installation

1. First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain
access to the model. For more information on user access tokens for access to gated models
see [here](https://huggingface.co/docs/hub/en/security-tokens).
2. Gain access to Llama3.2 set of models as well via this [link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on Hugging Face. 
3. Open `.env` file in this directory. Here you will find all variables which need to set in order to run the Video Summarizer. Default values have already been set.

```
# Please set Hugging Face access token for model access
HUGGINGFACE_TOKEN=

# Conda environment name, please change if you would like to use a different name
CONDA_ENV_NAME=ovlangvidsumm

# Summary Merger Tool endpoint
SUMMARY_MERGER_ENDPOINT="http://127.0.0.1:8000/merge_summaries"

# OVMS MiniCPM endpoint
OVMS_ENDPOINT="http://localhost:8013/v3/chat/completions"

# Device for the summary merger model: CPU, GPU or NPU
SUMMARY_MERGER_LLM_DEVICE="GPU"

# Device for the VLM model: CPU, GPU
VLM_DEVICE="GPU"

# Video summarization configuration
# Input video file, resolution, and prompt for summarization
INPUT_FILE="one-by-one-person-detection.mp4"
RESOLUTION_X=480
RESOLUTION_Y=270

PROMPT='As an expert investigator, please analyze this video. Summarize the video, highlighting any shoplifting or suspicious activity. The output must contain the following 3 sections: Overall Summary, Activity Observed, Potential Suspicious Activity. It should be formatted similar to the following example:

**Overall Summary**
Here is a detailed description of the video.

**Activity Observed**
1) Here is a bullet point list of the activities observed. If nothing is observed, say so, and the list should have no more than 10 items.

**Potential Suspicious Activity**
1) Here is a bullet point list of suspicious behavior (if any) to highlight.
'

# Parameters for summarization with --run_rag option
# Query text to search for in the Vector DB
# Example: "woman shoplifting"
QUERY_TEXT=
# Optional Filter expression for the Vector DB query
# Example: To search only text summaries: "mode=='text'"
FILTER_EXPR=
```

Next, run the Install script where installs all the dependencies needed.
```
# Validated on Ubuntu 24.04 and 22.04
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only, the following
command can be used to skip the re-install of dependencies. 

```
./install.sh --skip
```

Note: if running without --skip and the following message is produced: 
```
OpenVINO and OpenVINO Tokenizers versions are not binary compatible.
OpenVINO version:            2025.1.0-18503
OpenVINO Tokenizers version: 2025.1.0.0-523-710ddf14de8
First 3 numbers should be the same. Update OpenVINO Tokenizers to compatible version. It is recommended to use the same day builds for pre-release version. To install both OpenVINO and OpenVINO Tokenizers release version perform:
pip install --force-reinstall openvino openvino-tokenizers
To update both OpenVINO and OpenVINO Tokenizers to the latest pre-release version perform:
pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
Tokenizer won't be converted.
```

Please, run the following commands:
```
conda activate ovlangvidsumm
pip install -r requirements.txt
huggingface-cli login --token $HUGGINGFACE_TOKEN
rm -rf models
mkdir -p models
python export_model.py text_generation --source_model openbmb/MiniCPM-V-2_6 --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2 --pipeline_type VLM
```

## Convert and Save Optimized MiniCPM-V-2_6

This section can be skipped if you ran `install.sh` the first time. The `install.sh` script runs this command as part of 
its setup. This section is to give the user flexibility to tweak the `optimum-cli` command for certain model parameters. 

Ensure you `export HUGGINGFACE_TOKEN=<MY_TOKEN_HERE>` before executing the below command.
OR
Run `source .env` which will pick up the HUGGINGFACE_TOKEN variable from the file.
```
conda activate ovlangvidsumm
huggingface-cli login --token $HUGGINGFACE_TOKEN
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
mkdir -p models
python export_model.py text_generation --source_model openbmb/MiniCPM-V-2_6 --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2 --pipeline_type VLM
```

## Run Video Summarization

Summarize [this sample video](https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4)

```
./run-demo.sh 
```

Note: if the demo has already been run, you can use the following command to skip the video download.

```
./run-demo.sh --skip
```

## Run sample RAG application

Run RAG on the images and summaries you have ingested in the vector DB.

Open `run_demo.sh` and enter the QUERY_TEXT in `QUERY_TEXT=` variable. Then run the script.
```
./run_demo.sh --run_rag
```

## Milvus Setup

Milvus DB gets installed and setup when you run `install.sh`. 

To stop, start or delete the DB:

1. You can check the status of Milvus using the following command: `docker ps | grep milvus`

2. You can stop Milvus using the following command: `bash standalone_embed.sh start`
 
3. You can stop Milvus using the following command: `bash standalone_embed.sh stop`
 
4. You can delete Milvus data using the following command: `bash standalone_embed.sh delete`
