# langchain-openai-tts

This package contains the LangChain integration with OpenAIText2SpeechTool

## Installation

```bash
apt update && apt install -y portaudio19-dev build-essential ffmpeg
pip install -U .
```

## OpenAIText2SpeechTool
`OpenAIText2SpeechTool class exposes a tool for Text-To-Speech.

```python
from langchain_openai_tts import OpenAIText2SpeechTool
tts = OpenAIText2SpeechTool(
    model_id="kokoro", 
    voice="af_sky+af_bella", 
    base_url="http://localhost:8880/v1", 
    api_key="not-needed"
)
speech_file = tts.run(text_to_speak)
tts.play(speech_file)
# or stream
# tts.stream_speech(text_to_speak)
```
