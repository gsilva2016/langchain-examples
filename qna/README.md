$${\color{red}This \space sample \space  was \space  archived \space  by \space  the \space  owner \space  on \space  August \space  14th, \space  2025. \space  It \space  is \space  now \space  read-only.}$$


# QnA
Demonstrates a pipeline which performs QnA. The primary components utilize OpenVINO™ in LangChain for audio-speech-recognition, LLM text generation/response, and text-to-speech (currently [OuteAI/OuteTTS-0.1-350M](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/outetts-text-to-speech/outetts-text-to-speech.ipynb) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)).

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

### Audio QnA with RAG

This sample requires an audio file when DEMO_MODE=0. If DEMO_MODE=1 then audio must provided by the default connected microphone, otherwise an audio file must be provided as input. A sample wav file can be downloaded [here](https://github.com/intel/intel-extension-for-transformers/raw/refs/heads/main/intel_extension_for_transformers/neural_chat/assets/audio/sample_2.wav)

When utilizing RAG_ENABLED=1 the html/pdfs/etc must be stored in the ./docs folder. Files in this directory will automatically be loaded.

Run the below command to start the demo with the following defaults:

LLM Model: llmware/qwen2-0.5b-chat-ov<br>
LLM batch-size: 2<br>
ASR Model: distil-whisper/distil-small.en<br>
ASR load in 8bit: True<br>
ASR batch-size: 8<br>
TTS Model: kokoro<br>
RAG Enabled: 1<br>
DEMO MODE: 1<br>
Inference Device: GPU<br>

```
export LLM_MODEL="llmware/qwen2-0.5b-chat-ov"
#export LLM_MODEL="llmware/qwen2-1.5b-instruct-ov" # uncomment for better accuracy
export LLM_BATCH_SIZE=2
export ASR_MODEL=distil-whisper/distil-small.en
export ASR_LOAD_IN_8BIT=1
export ASR_BATCH_SIZE=8
export INF_DEVICE=GPU
export TTS_MODEL="kokoro"
export RAG_ENABLED=1
export DEMO_MODE=1
./run-audio-demo.sh myquestion.wav
```

### Text QnA with RAG

This sample requires a user to ask/type their question when prompted.  When utilizing RAG_ENABLED=1 the html/pdfs/etc must be stored in the ./docs folder. Files in this directory will automatically be loaded.

Run the below command to start the demo with the following defaults:

LLM Model: llmware/qwen2-0.5b-chat-ov<br>
LLM batch-size: 2<br>
RAG Enabled: 1<br>
DEMO MODE: 1<br>
Inference Device: GPU<br>

```
export LLM_MODEL="llmware/qwen2-0.5b-chat-ov"
#export LLM_MODEL="llmware/qwen2-1.5b-instruct-ov" # uncomment for better accuracy
export LLM_BATCH_SIZE=2
export INF_DEVICE=GPU
export RAG_ENABLED=1
export DEMO_MODE=1
./run-text-demo.sh
```

