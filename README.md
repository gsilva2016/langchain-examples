# LangChain* - Intel® GenAI Reference Samples

Various Intel® hardware and LangChain based examples are provided. Different parts of the workload can be distributed across edge, on-prem, or a CSP devices/infrastructure.

| Demo  | Description |
| ------------- | ------------- |
| [chapterization](chapterization) | Demonstrates an pipeline which automatically chapterizes long text/content from a provided audio context. The primary components utilize OpenVINO™ in LangChain* for audio-speech-recognition, embeddings generation, K-means clustering, and LLM chapterization.  |
| [qna](qna)  | Demonstrates a pipeline which performs QnA using audio or text with RAG. The primary components utilize OpenVINO™ in LangChain for audio-speech-recognition, LLM text generation/response, and text-to-speech.   |
| [video-summarization](video-summarization)  |  Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6.  |
| [eval-text-summarization-benchmarking](genai-eval-text-summarization-benchmarking)  |  Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated are BLEU, ROUGE-N, and BERTScore  |

Note: Please refer to following [guide](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/gpu-disable-hangcheck.html) for disabling GPU hangchecks.

FFmpeg is an open source project licensed under LGPL and GPL. See https://www.ffmpeg.org/legal.html. You are solely responsible for determining if your use of FFmpeg requires any additional licenses. Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of FFmpeg.
