# Diarized Sentiment via Audio Speech Recognition
Demonstrates a pipeline which performs Audio-Speech-Recognition (ASR), Diarization, and Sentiment Analysis. The primary components utilize PyTorch, OpenVINO™ and OpenVINO™ Model Server in LangChain.

## Installation

Get started by running the below command.

```
./install.sh
```

Note: if this script has already been performed and you'd like to install code change only then the below command can be used instead to skip the re-install of dependencies.

```
./install.sh --skip
```

```
./install-whisper.sh
```

```
./run-whisper.sh
```

## Run Examples

All of the samples utilize [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization and require the user to accept the [terms and conditions](https://huggingface.co/pyannote/speaker-diarization-3.1).  After accepting the terms be sure to set your HF ACCESS TOKEN in the .env file.

### Sentiment Analysis (text based) via audio file (non-live streaming)

This sample requires an audio file (wav/mp3). A sample wav file can be downloaded [here](https://github.com/intel/intel-extension-for-transformers/raw/refs/heads/main/intel_extension_for_transformers/neural_chat/assets/audio/sample_2.wav) or an audio file can be recorded using the Streamlit Demo.

#### Start OVMS (for Sentiment Analysis)
```
./run-ovms.sh
```

#### Start Diarized Sentiment Analysis Streamlit Demo (text based sentiment) via audio file
```
./run-streamlit-demo.sh
```

### Sentiment Analysis (text based) via live audio streaming

Coming soon

### Sentiment Analysis (wave based) via audio file (non-live streaming)

Coming soon

### Sentiment Analysis (wave based) via live audio streaming

Coming soon

