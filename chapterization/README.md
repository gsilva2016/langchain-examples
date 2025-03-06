# Chapterization
Demonstrates a pipeline which automatically chapterizes long text/content from a provided audio context. The primary components utilize OpenVINO™ in LangChain* for audio-speech-recognition, embeddings generation, K-means clustering, and LLM chapterization.

## Installation

Get started by running the below command.

```
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only then the below command can be used to skip the re-install of dependencies.

```
./install.sh --skip
```

## Run Examples

Run the below command to start the demo with the following defaults:

LLM Model: llmware/llama-3.2-3b-instruct-ov<br>
LLM batch-size: 2<br>
ASR Model: distil-whisper/distil-small.en<br>
ASR load in 8bit: True<br>
ASR batch-size: 8<br>
Inference Device: GPU<br>
K-Means Clustering Enabled For Text : True<br>

```
export LLM_MODEL=llmware/llama-3.2-3b-instruct-ov
export LLM_BATCH_SIZE=2
export ASR_MODEL=distil-whisper/distil-small.en
export ASR_LOAD_IN_8BIT=1
export ASR_BATCH_SIZE=8
export INF_DEVICE=GPU
export ENABLE_KMEANS=1
./run-demo.sh audio.mp3
```
