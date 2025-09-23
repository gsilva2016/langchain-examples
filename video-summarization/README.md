# Video Analysis Pipeline using Person Tracking/ReID, Summarization using GenAI and traditional CV techniques

Uses: OpenVINO Model Server, open source/custom Langchain packages, MiniCPM-V-2_6, Llama3.2-3B, DeepSORT/ReID and Milvus.

## Introduction

This repository provides a modular pipeline for video analysis and summarization, combining CV and genAI techniques. The main modules include:

- **Person Tracking & Re-Identification (ReID):** Uses DeepSORT and ReID models to detect, track, and uniquely identify individuals across video frames. Uses Milvus to reconcile across same/multiple inputs.
- **Summarization:** Employs vision-language models (MiniCPM-V-2_6) and large language models (Llama-3.2-3B) to generate concise, human-readable summaries of video content.
- **Embedding & Vector Database:** Utilizes BLIP for generating embeddings from text and images, storing them in Milvus for efficient similarity search and retrieval.
- **Vector Search/Retriever** Enables search and retrieval over ingested video data using both text and image queries.
- **Model Serving:** Integrates OpenVINO Model Server (OVMS) for efficient model inference and deployment.

Each module can be enabled or disabled independently, allowing flexible experimentation and customization for different video analytics scenarios. (Refer to the [Run Video Pipeline](#run-video-pipeline) section.)

## System Requirements

•	CPU: Intel Core Platform (tested on Arrow Lake, Lunar Lake, and Meteor Lake platforms)

•	GPU: iGPU required for accelerated runtimes. If using Intel discrete GPU, change to appropriate GPU ID in the .env file

•	Memory: >= 32G

•	OS: Ubuntu 24.04

## Installation and .env (environment variable file) Configuration

1. First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain
access to the model. For more information on user access tokens for access to gated models
see [here](https://huggingface.co/docs/hub/en/security-tokens).

2. Gain access to [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) group of models on Hugging Face. 

3. Next, open `.env` file in this current directory. Here you will find all the variables which need to set in order to run the Video Summarizer. Default values have already been set.

4. The tracking/ReID modules require finetuning the DeepSORT configuration parameters as per the input video/camera stream being tested.
    
    For the DeepSORT module, please finetune these values in the .env.

    ```
    TRACKER_NN_BUDGET=100
    TRACKER_MAX_COSINE_DISTANCE=0.3
    TRACKER_MAX_IOU_DISTANCE=0.7
    TRACKER_MAX_AGE=100
    TRACKER_N_INIT=5
    TRACKER_DET_THRESH=0.7
    ```

    For example:

    TRACKER_NN_BUDGET=100

    Maximum number of feature samples to keep for each track. Higher values make tracking more stable and robust to appearance changes, but increase memory usage and may slow down matching. Lower values make tracking more responsive to appearance changes, but tracks may switch IDs more easily or be lost in challenging scenarios. Feature samples can be updated per frame and because the tracker processes every frame in the input video by default, this optimal value depends on the input framerate. The default setting has been calibrated for the standard input video included with this application.

    TRACKER_MAX_COSINE_DISTANCE=0.3
    
    Controls how strictly reidentification features are matched. One of the most powerful levers for reducing ID switches. Lower values = stricter, fewer ID switches, but may lose tracks if appearance changes. Higher values = more flexible, but may increase ID switches or false matches. 
    
    TRACKER_MAX_IOU_DISTANCE=0.7

    Controls how strictly detections are associated with existing tracks. Lower values = stricter, tracks break more easily if detections shift. Higher values = more flexible, but may increase false positives or ID switches. 

    TRACKER_MAX_AGE=100

    Maximum number of frames to keep a track alive without a matching detection. Higher values = tracks persist longer through occlusions/missed detections, but may increase false positives. Lower values = lost tracks removed quickly, but may lose tracks during short occlusions. This parameter usually requires the most fine-tuning. Because the tracker processes every frame in the input video by default, its optimal value depends on the input framerate. The default setting has been calibrated for the standard input video included with this application.

    TRACKER_N_INIT=5

    Number of consecutive matches before confirming a track. Higher values = more consistent detection required (reduces false positives, but may delay/miss true tracks). Lower values = tracks confirmed quickly, but may allow more false tracks.

    TRACKER_DET_THRESH=0.7
    
    Detection threshold for the person detector. Usually does not require finetuning but may be required if people are poorly lit / not easily visible.

    
    For the ReID reconcillation module, please finetune these three values in the .env.
    
    ```
    REID_SIM_SCORE_THRESHOLD=0.67
    TOO_SIMILAR_THRESHOLD=0.96
    AMBIGUITY_MARGIN=0.02
    ```

    For example:
    
    REID_SIM_SCORE_THRESHOLD=0.67. 
    
    This value will need the most finetuning. 0.67 seems to work well for the standard input video provided in this app. Typical values will range from 0.56 to 0.9. 
    This value indicates what similarity score threshold value defines if a person detected is a new person or not. 
    Ex: If Frame 2 with a person returns a similarity score of 0.7, then it means that the person was already seen previously and is not a new person. 
    If the similarity score is less than 0.67, its a new person detected and their embedding gets stored in Milvus database for future comparisons/references.

    
    TOO_SIMILAR_THRESHOLD=0.96
    
    This value most likely would not need any finetuning and can remain the same. To prevent database bloat and prevent too many similar entries being stored in the vector DB, 
    this value indicates that if the similarity score is greater than 0.96, then we don't need to store its embedding in the DB since we alrady have an embedding which is representative.

    AMBIGUITY_MARGIN=0.02
    
    This value has been put in place to ignore certain embeddings which are ambiguous and very close to the REID_SIM_SCORE_THRESHOLD and sometimes can result in false positive. 
    A value of 0.02 is interpreted as follows:
    
    This skips embeddings whose similarity score is in a "borderline" range. Specifically, if the similarity score is just below the main threshold (SIM_SCORE_THRESHOLD) but within a margin (AMBIGUITY_MARGIN).

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
TRACKER_DEVICE="GPU"
TRACKER_NN_BUDGET=100
TRACKER_MAX_COSINE_DISTANCE=0.3
TRACKER_METRIC_TYPE="cosine"
TRACKER_MAX_IOU_DISTANCE=0.7
TRACKER_MAX_AGE=100
TRACKER_N_INIT=5
TRACKER_WIDTH=700
TRACKER_HEIGHT=450
TRACKER_DET_THRESH=0.7

####### Parameters for Milvus

# Default connection parameters for Milvus
MILVUS_HOST="localhost"
MILVUS_PORT=19530
MILVUS_DBNAME="default"

# Names of the Milvus collections for storing video chunks, re-identification data, and tracking logs
VIDEO_COLLECTION_NAME="video_chunks"
REID_COLLECTION_NAME="reid_data"
TRACKING_COLLECTION_NAME="tracking_logs"

# Similarity thresholds for re-identification and duplicate detection (detailed explanations in the README)
REID_SIM_SCORE_THRESHOLD=0.67
TOO_SIMILAR_THRESHOLD=0.96
AMBIGUITY_MARGIN=0.02

# Interval for creating partitions in Milvus (in hours). If a partition for the current hour doesn't exist, it will be created.
# This helps in organizing data and improving query performance.
PARTITION_CREATION_INTERVAL=1 # in hours
# Interval for flushing tracking logs to Milvus (in seconds). 
# Data will be flushed to Milvus at this interval to ensure it's saved and available for queries.
TRACKING_LOGS_GENERATION_TIME_SECS=1
# If True, overwrites the existing Milvus collection. Use with caution as this will delete existing data. Useful for testing.
OVERWRITE_MILVUS_COLLECTION="TRUE"

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

## Run sample vector search/retrieve application

Run vector search/retrieve on the images and summaries you have ingested in the vector DB.

*Note*: Currently vector search/retrieve and video-summarization run as separate processes. Please run searches on input videos you have already run video-summarization on. 

3 types of vector search/retrieve are possible currently:

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

## Run Idle Agent

To enable Idle Agent flow, which periodically updates the idle status and stores it along with a short summary in MilvusDB, run the following command:
```
./run_demo.sh --run_agent_monitor
```
Note: Ensure that MilvusDB is properly set up and contains the necesseray data before running the agent.