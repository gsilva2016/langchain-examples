# langchain-summarymerge-score

This package contains the LangChain integration with SummaryMergeScore

## Installation

```bash
pip install -U langchain-summarymerge-score
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSummaryMergeScore` class exposes chat models from SummaryMergeScore.

```python
from langchain_summarymerge_score import ChatSummaryMergeScore

llm = ChatSummaryMergeScore()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SummaryMergeScoreEmbeddings` class exposes embeddings from SummaryMergeScore.

```python
from langchain_summarymerge_score import SummaryMergeScoreEmbeddings

embeddings = SummaryMergeScoreEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SummaryMergeScoreLLM` class exposes LLMs from SummaryMergeScore.

```python
from langchain_summarymerge_score import SummaryMergeScoreLLM

llm = SummaryMergeScoreLLM()
llm.invoke("The meaning of life is")
```
