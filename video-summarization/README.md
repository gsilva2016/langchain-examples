# Summarize Videos Using OpenVINO Model Server, Langchain, and MiniCPM-V-2_6

## Installation

1. First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain
access to the model. For more information on user access tokens for access to gated models
see [here](https://huggingface.co/docs/hub/en/security-tokens).

2. Gain access to [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) group of models on Hugging Face. 

3. Next, open `.env` file in this current directory. Here you will find all the variables which need to set in order to run the Video Summarizer. Default values have already been set.

```
# Hugging Face access token for model access
HUGGINGFACE_TOKEN=

# Conda environment name, please change if you would like to use a different name
CONDA_ENV_NAME=ovlangvidsumm

# OVMS endpoint for all models
OVMS_CONDA_ENV_NAME=ovms_env
OVMS_ENDPOINT="http://localhost:8013/v3/chat/completions"

####### Sub Module Status
# If False, Doesn't run MiniCPM, LLAMA pipelines. 
RUN_VLM_PIPELINE="TRUE"
# If False, Doesn't run Tracking pipeline. 
RUN_REID_PIPELINE="TRUE"
# If True, saves re-identification visualization videos in chunks directory
SAVE_REID_VIZ_VIDOES="FALSE"

####### Video ingestion configuration
OBJ_DETECT_ENABLED="FALSE"
OBJ_DETECT_MODEL_PATH="ov_dfine/dfine-s-coco.xml"
OBJ_DETECT_SAMPLE_RATE=5
OBJ_DETECT_THRESHOLD=0.9

####### Summary merger configuration

# Name of the LLM model for summary merging in Hugging Face format (model runs on OVMS model server)
LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# Device for the summary merger model: CPU, GPU or NPU
SUMMARY_MERGER_LLM_DEVICE="GPU"

# Prompt for merging multiple chunk summaries into one summary
SUMMARY_PROMPT='You are a video summarization agent. Your job is to merge multiple chunk summaries into one balanced and complete summary.
Important: Treat all summaries as equally important. Do not prioritize summaries that have more detail or mention suspicious activity - your goal is to combine information, not amplify it.

Guidelines:
- Extract the most important insights from all summaries into a concise output.
- Give equal importance to each chunk, regardless of its length or uniqueness.
- Estimate the number of people in the scene based on textual hints. If summaries mention no people or say "empty" or "no visible customers", count that as 0 people. If someone is mentioned (e.g., "a customer", "a person walks in"), count them.
- Only output one summary in the exact format below.
- Do not include the input summaries or instructions in your response.

Output Format:
Overall Summary: <summary here>
Activity Observed: <bullet points>
Potential Suspicious Activity: <suspicious behavior if any>
Number of people in the scene: <best estimate. DO NOT overcount>.
Anomaly Score: <float from 0 to 1, based on severity of suspicious activity>.
'

####### Embedding model configuration
# Currently verified model
EMBEDDING_MODEL="Salesforce/blip-itm-base-coco"

# Device for text embeddings: CPU, GPU
# Important Note: If you are changing the device here, please make sure to delete the old blip model files (bin and xml files in current dir)
TXT_EMBEDDING_DEVICE="GPU"

# Device for img embeddings: CPU, GPU, NPU 
# Important Note: If you are changing the device here, please make sure to delete the old blip model files (bin and xml files in current dir)
IMG_EMBEDDING_DEVICE="GPU"

####### Video summarization configuration
# Input video file, resolution, and prompt for summarization

# VLM model
VLM_MODEL="openbmb/MiniCPM-V-2_6"

# Device for the VLM model: CPU, GPU
VLM_DEVICE="GPU"

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

####### DeepSORT tracker configuration

TRACKER_DET_MODEL_PATH="tracker_models/person-detection-0202/FP16/person-detection-0202.xml"
TRACKER_REID_MODEL_PATH="tracker_models/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml"
TRACKER_DEVICE="AUTO"
TRACKER_NN_BUDGET=100
TRACKER_MAX_COSINE_DISTANCE=0.5
TRACKER_METRIC_TYPE="cosine"
TRACKER_MAX_IOU_DISTANCE=0.7
TRACKER_MAX_AGE=100
TRACKER_N_INIT=1
TRACKER_WIDTH=700
TRACKER_HEIGHT=450
TRACKER_DET_THRESH=0.5

####### Parameters for Milvus

MILVUS_HOST="localhost"
MILVUS_PORT=19530
MILVUS_DBNAME="default"
VIDEO_COLLECTION_NAME="video_chunks"
REID_COLLECTION_NAME="reid_data"
TRACKING_COLLECTION_NAME="tracking_logs"
REID_SIM_SCORE_THRESHOLD=0.7
TOO_SIMILAR_THRESHOLD=0.99

####### Parameters for reporting agent

# Time in seconds
REPORT_GENERATION_TIME_SECS=10  

####### Parameters for summarization with --run_rag option

# Query text to search for in the Vector DB
# Example: "woman shoplifting"
QUERY_TEXT=
# Query image to search for in the Vector DB (path to the image file)
QUERY_IMG=

# Optional Filter expression for the Vector DB query
# Filters can be used to narrow down the search results based on specific criteria.
# The filter expression can include:
# - mode: 'text' or 'image' (whether to search using the text summary embeddings or frame embeddings)
# - video_path: path to the video file. Search will be limited to chunks from this video.
# - chunk_path: path to the chunk file. Search will be limited to specific chunk file.
# - detected_objects: list of objects detected in the chunk, e.g., 'person', 'car', 'bag', etc. An example is provided below. 

# Examples of various types of `FILTER_EXPR` is provided below. Various supported operators can be referred to in Milvus' documentation - https://milvus.io/docs/boolean.md 
# Example: To search only text summaries: "metadata['mode'] == 'text'". To search only frames: "metadata['mode'] == 'image'"
# Example: To search for specific detected objects: "metadata['detected_objects'] LIKE '%<object name>%'"
# Example: To search on a specific video: "metadata['video_path'] == '<path to video>'"
# Example: Combine multiple filters using operator "and": "metadata['mode'] == 'image' and metadata['video_path'] == '<path to video>' and metadata['detected_objects'] LIKE '%person%'"

### Example: FILTER_EXPR="metadata['mode'] == 'image' and metadata['detected_objects'] LIKE '%person%'". Search includes only image embeddings with detected objects that contain 'person'.
# Example: FILTER_EXPR="metadata['mode'] == 'text'" for searching through summaries
FILTER_EXPR=

# Save a video clip of Milvus search result
SAVE_VIDEO_CLIP=True
# Duration of the video clip in seconds
VIDEO_CLIP_DURATION=7
# Number of top results to retrieve from Milvus
RETRIEVE_TOP_K=1

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

## Convert and Save Optimized MiniCPM-V-2_6 VLM and Llama-3.2-3B LLM

This section can be skipped if you ran `install.sh` the first time. The `install.sh` script runs this command as part of 
its setup. This section is to give the user flexibility to tweak the `export_model.py` command for certain model parameters to run on OVMS.

Ensure you `export HUGGINGFACE_TOKEN=<MY_TOKEN_HERE>` before executing the below command.

OR

Run `source .env` which will pick up the HUGGINGFACE_TOKEN variable from the file.

```
conda activate ovlangvidsumm

huggingface-cli login --token $HUGGINGFACE_TOKEN

curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py

mkdir -p models

1. # export miniCPM model on GPU
python export_model.py text_generation --source_model openbmb/MiniCPM-V-2_6 --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2 --pipeline_type VLM

2. # export LLAMA3.2 model on GPU
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2 --pipeline_type LM --overwrite_models

OR 

# export LLAMA3.2 model on NPU
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --config_file_path models/config.json --model_repository_path models --target_device NPU --max_prompt_len 1500 --pipeline_type LM --overwrite_models
```

## Model Server

Pipeline uses [OVMS (OpenVINO Model Server)](https://github.com/openvinotoolkit/model_server) for serving the VLM and LLM.

## Embedding Model

This pipeline uses BLIP for creating embeddings out of text and images. The subsequent vectors are stored in MIlvus DB. Model details are present in the configuration file `.env`.

## Run Video Pipeline

The Video Pipeline consists of:

1. Tracking/REID modules

2. Summarization modules (miniCPM and LLAMA based pipeline)

To turn off the Summarization pipeline, set RUN_VLM_PIPELINE="FALSE" in .env file. Default is TRUE.

To turn off the Tracking pipeline, set RUN_REID_PIPELINE="FALSE" in .env file. Default is TRUE.

The Tracking module can save videos with the tracking results overlayed. To turn it on, set SAVE_REID_VIZ_VIDOES="TRUE" in .env file. Default is FALSE.

Run Video Pipeline on [this sample video](https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4)

```
./run-demo.sh 
```

Note: if the demo has already been run, you can use the following command to skip the video download.

```
./run-demo.sh --skip
```

## Run sample RAG application

Run RAG on the images and summaries you have ingested in the vector DB.

*Note*: Currently RAG and video-summarization run as separate processes. Please run RAG searches on input videos you have already run video-summarization on. 

3 types of RAG searches are possible currently:

1. Text based similarity search
Open `run_demo.sh` and enter the QUERY_TEXT in `QUERY_TEXT=` and `FILTER_EXPR`(optional) variable. 

2. Image based similarity search
Open `run_demo.sh` and enter the QUERY_IMG (path to image) in `QUERY_TEXT=` and `FILTER_EXPR`(optional) variable. 

3. Query based on certain filter expressions (no text or image)
Open `run_demo.sh` and enter the `FILTER_EXPR` variable. Then run the script.

Examples of various types of `FILTER_EXPR` that can be created is annotated in `run_demo.sh`. Various suported operators can be referred to in Milvus' documentation [here](https://milvus.io/docs/boolean.md) 

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
