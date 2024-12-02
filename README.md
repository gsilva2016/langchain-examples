# LangChain Intel AI PC Examples

Demo-1 is meant to demonstrate running LangChain based applications using hybrid architectures. Demo-1 can be run on an Intel AI PC and distribute pieces of the pipeline across on-prem edge servers and/or CSP managed AI services. Demo-1 (run-demo-1.sh) will perform three tests to demonstrate this with LangChain.

Test #1
* First perform direct OpenVINO Model Server (OVMS) embeddings call from the specified EMBEDDINGS_API_ENDPOINT for model BAAI/bge-small-en. The default endpoint is configured for localhost which assumes the OVMS server is ran locally (if not then update the endpoint accordingly).
* Second, load ./docs into FAISS vector store and query the vector store using the retriver.
* Third, query the GPT2 LLM using OpenVINO.
* Lastly, perform RAG query with the GPT2 LLM and the vector store results.
* All inference is performed on CPU.

Test #2
* First perform direct OpenVINO Model Server (OVMS) embeddings call from the specified EMBEDDINGS_API_ENDPOINT for model sentence-transformers/all-mpnet-base-v2. The default endpoint is configured for localhost which assumes the OVMS server is ran locally (if not then update the endpoint accordingly).
* Second, load ./docs into FAISS vector store and query the vector store using the retriver.
* Third, query the llmware/qwen2.5-1.5b-instruct-ov LLM using OpenVINO.
* Lastly, perform RAG query with the llmware/qwen2.5-1.5b-instruct-ov LLM and the vector store results.
* All inference is performed on the integrated GPU (iGPU).

Test #3
* First perform direct AWS Bedrock embeddings call for the model amazon.titan-embed-text-v2. 
* Second, load ./docs into FAISS vector store and query the vector store using the retriver.
* Third, query the llmware/qwen2.5-1.5b-instruct-ov LLM using OpenVINO.
* Lastly, perform RAG query with the llmware/qwen2.5-1.5b-instruct-ov LLM and vector store results.
* All inference is performed on the integrated GPU (iGPU) except the embeddings generation performed on AWS Bedrock.


## Installation

Get started by running the below.

```
./install.sh
```

## Running Demos

Start Intel OpenVINO Model Server for GenAI Serving by running the below.

```
./run-ovms-llm-serving.sh
```

Start the demo-1 by running the below.

```
./run-demo-1.sh
````
