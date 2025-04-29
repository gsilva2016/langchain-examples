# langchain-openvino-asr

This package contains the LangChain integration with OpenVINOSpeechToTextLoader

## Installation

```bash
pip install .
```

## OpenVINOSpeechToTextLoader

`OpenVINOSpeechToTextLoader` class exposes a document loader for Speech-To-Text.

```python
from langchain_openvino_asr import OpenVINOSpeechToTextLoader

loader = OpenVINOSpeechToTextLoader(
    file_path = "./audio.wav",
    model_id = "distil-whisper/distil-small.en",
    # optional params
    # device = "CPU", # GPU
    # return_timestamps = True,
    # return_language = "en",
    # chunk_length_s = 30,
    # load_in_8bit = True,
    # batch_size = 2,
)

docs = loader.load()
```
