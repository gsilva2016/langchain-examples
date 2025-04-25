# langchain-summarymerge-score

This package contains the LangChain integration with SummaryMergeScore

## Installation

```bash
pip install -U langchain-summarymerge-score
```

And you should configure credentials by setting the following environment variables:

* Set HF_ACCESS_TOKEN via `export HF_ACCESS_TOKEN=<YOUR_ACCESS_TOKEN>`

## Using the tool (SummaryMergeScore) via local endpoint server
The SummaryMergeScore tool is also available via a local FastAPI server. 

To use the tool via local FastAPI endpoints:

1. Ensure your endpoint server is up and running (example: a FastAPI based server hosting your model of choice). Lets say it is hosted at: `http://localhost:8000/merge_summaries`

2. Invoke the tool via:

```python
from langchain_summarymerge_score import SummaryMergeScoreTool

summary_merger = SummaryMergeScoreTool(
    api_base="http://localhost:8000/merge_summaries"
)

summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

output = summary_merger.invoke({"summaries": summaries})
```

## Using the tool (SummaryMergeScore) via tool invokation (without FastAPI endpoint server)

```python
from langchain_summarymerge_score import SummaryMergeScoreTool

summary_merger = SummaryMergeScoreTool(
    model_id="llmware/llama-3.2-3b-instruct-ov",
    device="GPU"
)

summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

output = summary_merger.invoke({"summaries": summaries})
```
