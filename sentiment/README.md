# QnA
Demonstrates a pipeline which performs Audio-Speech-Recognition (ASR), Diarization, and Sentiment Analysis. The primary components utilize OpenVINO™ and OpenVINO™ Model Server in LangChain.

## Installation

Get started by running the below command.

```
./install.sh
```

Note: if this script has already been performed and you'd like to install code change only then the below command can be used instead to skip the re-install of dependencies.

```
./install.sh --skip
```

## Run Examples

All of the samples utilize [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization and require the user to accept the [terms and conditions](https://huggingface.co/pyannote/speaker-diarization-3.1).  After accepting the terms be sure to set your HF ACCESS TOKEN via `export HF_ACCESS_TOKEN=<YOUR_TOKEN_HERE>`

### Sentiment Analysis (text based) via audio file (non-live streaming)

This sample requires an audio file and DEMO_MODE=0 to be set. If DEMO_MODE=1 then audio must provided by the default connected microphone (unsupported at this time). A sample wav file can be downloaded [here](https://github.com/intel/intel-extension-for-transformers/raw/refs/heads/main/intel_extension_for_transformers/neural_chat/assets/audio/sample_2.wav)

Run the below command to start the demo with the following defaults:

Sentiment Model: llmware/qwen2-0.5b-chat-ov<br>
Demo Mode: 0<br>
Inference Device: GPU<br>

```
#export SENTIMENT_MODEL="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4" # uncomment for remote inference
#export SENTIMENT_MODEL="llmware/qwen2-0.5b-chat-ov" # uncomment for faster inference
export SENTIMENT_MODEL="llmware/qwen2-1.5b-instruct-ov" # uncomment for better accuracy
export INF_DEVICE=GPU
export DEMO_MODE=0
./run-demo.sh audio.mp3
```

### Sentiment Analysis (text based) via live audio streaming

Coming soon

### Sentiment Analysis (wave based) via audio file (non-live streaming)

Coming soon

### Sentiment Analysis (wave based) via live audio streaming

Coming soon
