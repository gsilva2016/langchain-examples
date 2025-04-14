# LangChain* - Intel® GenAI Reference Samples

Various Intel® hardware and LangChain based examples are provided. Different parts of the workload can be distributed across edge, on-prem, or a CSP devices/infrastructure.

| Demo  | Description |
| ------------- | ------------- |
| [chapterization](chapterization) | Demonstrates an pipeline which automatically chapterizes long text/content from a provided audio context. The primary components utilize OpenVINO™ in LangChain* for audio-speech-recognition, embeddings generation, K-means clustering, and LLM chapterization.  |
| [qna](qna)  | Demonstrates a pipeline which performs QnA using audio or text with RAG. The primary components utilize OpenVINO™ in LangChain for audio-speech-recognition, LLM text generation/response, and text-to-speech.   |
| [video-summarization](video-summarization)  |  Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6.  |
| [eval-text-summarization-benchmarking](genai-eval-text-summarization-benchmarking)  |  Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated are BLEU, ROUGE-N, and BERTScore  |

<b>Note:</b> Please refer to following [guide](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/gpu-disable-hangcheck.html) for disabling GPU hangchecks.

<b>FFmpeg</b>

FFmpeg is an open source project licensed under LGPL and GPL. See https://www.ffmpeg.org/legal.html. You are solely responsible for determining if your use of FFmpeg requires any additional licenses. Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of FFmpeg.

<b>AI Model Usage</b>

This repository utilizes various AI models, including fully open-source models (e.g., Apache 2.0, MIT-licensed models) and openly available models (e.g., Meta’s LLaMA). Each model remains the intellectual property of its respective creators and is governed by its original license. Fully open-source models such as Qwen, Mistral, and Phi allow unrestricted use, modification, and distribution under their respective licenses. Models like LLaMA are available under specific usage terms that may include restrictions. This document does not redistribute, modify, or alter any model weights, nor does it claim ownership over them. Users should consult the official licensing terms of each model before use. The inclusion of specific models in this whitepaper does not imply endorsement by their respective developers. The recommendations presented are based on independent research and do not constitute official guidance from Meta, OpenAI, Google, or any other model provider. 
