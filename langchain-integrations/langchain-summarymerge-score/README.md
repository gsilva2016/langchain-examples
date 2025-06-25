# langchain-summarymerge-score

This package contains the LangChain integration with SummaryMergeScore

## Installation

```bash
pip install -U langchain-summarymerge-score
```

And you should configure credentials by setting the following environment variables:

* Set HUGGINGFACE_TOKEN via `export HUGGINGFACE_TOKEN=<YOUR_ACCESS_TOKEN>`

## Using the tool (SummaryMergeScore) via OVMS endpoint server
The SummaryMergeScore tool expects the LLM model to be available via one of two ways:

1. OVMS model server 

2. Local openVINO model directory (obtained through [`optimum-cli`](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html) command)

To use the tool via an OVMS model server (local or remote):

1. Ensure your model server is up and running. Ex: http://localhost:8013/v3/chat/completions

2. Invoke the tool via:

```python
from langchain_summarymerge_score import SummaryMergeScoreTool

tool = SummaryMergeScoreTool(
    api_base="http://localhost:8013/v3/chat/completions", # your OVMS endpoint
    device="GPU", # CPU, GPU, or NPU
    max_new_tokens=512,
    batch_size=5,
)

summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

output = tool.invoke({"summaries": summaries})

# output will look like:
{"overall_summary": "Merged summary text", "anomaly_score": 0.5}    
```

## Using the tool (SummaryMergeScore) via local model directory
```python
from langchain_summarymerge_score import SummaryMergeScoreTool

tool = SummaryMergeScoreTool(
    model_id="my_meta_llama_3.2_3B_Instruct/", # path to local openVINO model directory, ensure dir points to OV IR files
    device="GPU", # CPU, GPU, or NPU
    max_new_tokens=512,
    batch_size=5,
)

summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

output = tool.invoke({"summaries": summaries})

# output will look like:
{"overall_summary": "Merged summary text", "anomaly_score": 0.5}    
```
