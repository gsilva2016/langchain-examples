import argparse
import time
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLMOpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_aws import BedrockEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import docs_loader_utils as docs_loader
import agent_builder as lanchain_agent_builder
from pymilvus import Collection
from pymilvus import connections
from pymilvus.orm import utility
from pymilvus import db, Collection
from langchain_milvus import Milvus
from pymilvus import Collection, connections
from langchain_core.documents import Document
from datetime import datetime

# OVMS in LangChain Reference: https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb

# OVMS_EMBEDDING_API_ENDPOINT
EMBEDDINGS_API_ENDPOINT = "http://localhost:8000/v3/embeddings"
EMBEDDINGS_API_KEY = "EMPTY"
MAX_NEW_TOKENS = 500

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", nargs="?", default="gpt2")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--q", nargs="?", default="Explain how to teach math to third grade students.")
parser.add_argument("--embeddings_api_endpoint", nargs="?", default=EMBEDDINGS_API_ENDPOINT)
# OV LangChain: sentence-transformers/all-mpnet-base-v2
# OPEA: bge-large-zh-v1.5
parser.add_argument("--embeddings_model", nargs="?", default="BAAI/bge-small-en")
parser.add_argument("--skip_docs_loading", default=False, action="store_true")
args = parser.parse_args()

# Print Args
print("LangChain OpenVINO LLM - Model Serving Hybrid Model Swap Demo")
print("This demonstrates LangChain usage with OpenVINO LLM model swapping in a hybrid architecture via OpenVINO Model Server")
print("LLM model_id: ", args.model_id)
print("LLM device  : ", args.device)
print("LLM question: ", args.q)
print("llm_api: N/A - localhost")
print("embeddings_api_endpoint: ", args.embeddings_api_endpoint if not "amazon" in args.embeddings_model else "todo")
print("embeddings_model", args.embeddings_model)
print("rag document skip loading: ", args.skip_docs_loading)
print("rag documents download dir: ./docs")
#print("vector store: FAISS")
input("Press Enter to continue...")
# 

# Embeddings Gen. -- non-local (is served via vLLM OpenVINO Model Server (OVMS) via LangChain
# edge/on-prem/CSP examples via OVMS and vLLM
# LangChain OpenVINO Model Serving via OPENAPI compatible API - https://github.com/openvinotoolkit/model_server/tree/69e232d153336c2422b5a91373498166117656c0/demos/continuous_batching/rag
if "amazon" in args.embeddings_model:
    # CSP managed service example via Bedrock - requires an AWS Bedrock account
    embeddings = BedrockEmbeddings(
        credentials_profile_name="bedrock-admin", 
        region_name="us-west-1",
        model_id=args.embeddings_model
        )
else:
    # AI PC / on-prem server / CSP
    embeddings = OpenAIEmbeddings(
        api_key=EMBEDDINGS_API_KEY,
        base_url=args.embeddings_api_endpoint,
        model=args.embeddings_model,
        tiktoken_enabled=False,
        embedding_ctx_length=8190
        )
        # ChatOpenAI can utilize VLLM as a model server
        # Alternatively VLLMOpenAI can be used directly as shown below
        #embeddings = VLLMOpenAI(
        #        openai_api_key=EMBEDDINGS_API_KEY,
        #        openai_api_base=args.embeddings_api_endpoint,
        #        model_name=args.embeddings_model,
        #        default_headers={},
        #        max_tokens=MAX_NEW_TOKENS,
        #        top_p=0.60,
        #        top_k=1,
        #        streaming=True,
        #        temperature=0.1
        #        )

print("Starting EMBEDDINGS_API inference to OVMS via LangChain")
embs_res = embeddings.embed_query(args.q)
print("Embeddings result (100 chars only for brevity): ", str(embs_res)[:100])
input("Press Enter to continue...")

# RAG - load docs
print("Utilize chosen local/remote embeddings service with a local/remote milvus vector store")
# Connect to Milvus
conn = connections.connect(host="127.0.0.1", port=19530)
if "milvus_db" not in db.list_database():
    db.create_database("milvus_db")

db.using_database("milvus_db")
collections = utility.list_collections()
        
for name in collections:
    # Collection(name).drop()
    print(f"Collection {name} exists.")

# Create Milvus instance
db = Milvus(
    embedding_function=embeddings, # old method used local memory via ov_txt_embeddings. With langchain it is plugnplay with remote as well
    collection_name="chunk_summaries",
    connection_args={"uri": "http://127.0.0.1:19530", "db_name": "milvus_db"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    drop_old=False,
)
print("Connected to Milvus DB successfully.")
documents = [
    Document(
        page_content="Something is occuring in this chunk",
        metadata={
            "camera_id": "cam1",
            "chunk_id": 1,
            "start_time": "0",
            "end_time": "5",
            "chunk_path": "/tmp/1.mp4",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    ),
    Document(
        page_content="Something else is occuring in this chunk",
        metadata={
            "camera_id": "cam2",
            "chunk_id": 1,
            "start_time": "0",
            "end_time": "5",
            "chunk_path": "/tmp/1.mp4",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
]

ids = [f"{doc.metadata['camera_id']}_{doc.metadata['chunk_id']}" for doc in documents]
print("ids to save to db: ", ids)
db.add_documents(documents=documents, ids=ids)
print( "total_chunks:", len(documents))


#db = docs_loader.load_docs(embeddings, args.skip_docs_loading)
vector_search_top_k = 5
# RAG - get retriever
retriever = db.as_retriever(search_kwargs={"k": vector_search_top_k})
# RAG - query vector store using the retriever
retrieved_docs = retriever.invoke(args.q)
print("Retriever results for the question (500 chars only for brevity): ")
docs_loader.pretty_print_docs(retrieved_docs)
input("Press Enter to continue...")

# Edge local LLM inference example 
print("Starting LLM inference...")
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": args.device, "ov_config": ov_config}, # used for invoke
    pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS},
)

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

# below max is used by streamer
generation_config = {"skip_prompt": True, "pipeline_kwargs": {"max_new_tokens": MAX_NEW_TOKENS}} 

chain = prompt | ov_llm.bind(**generation_config)

print("-------LLM Answer for question---------")
for chunk in chain.stream(args.q):
    print(chunk, end="", flush=True)
#print(ov_llm.invoke(question))
print("\n")
input("Press Enter to continue...")

# Finally RAG+Agent Tool Example using chains with swappable models and deployment architectures across edge, on-prem, csp
# NOTE: Tool calling omitted due to poor support
# Traceback (most recent call last):
# File "/home/upx/gsilv/langchain/demo-1.py", line 129, in <module>
#   model_with_tools = ov_llm.bind_tools(tools)
# File "/home/upx/gsilv/langchain/langchain_aicaps_demo_1_env/lib/python3.10/site-packages/pydantic/main.py", line 856, in __getattr__
#    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')
#AttributeError: 'HuggingFacePipeline' object has no attribute 'bind_tools'

print("Starting RAG Chain (tool calling omitted)...")

# Tool calling
#tools = [lanchain_agent_builder.local_db_knowledge_base]
#model_with_tools = ov_llm.bind_tools(tools)
# Alternative native approach is possible: https://docs.openvino.ai/2024/notebooks/llm-agent-react-with-output.html#create-tool-calling

prompt = PromptTemplate(
    input_variables=['context', 'question'], 
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.

    Question: {question} \nContext: {context} \nAnswer:")"""
    )
# TODO: https://discuss.huggingface.co/t/how-to-implement-bind-tools-to-custom-llm-from-huggingface-pipeline-llama-3-for-a-custom-agent/95545
rag_chain = (
    {"context": retriever | docs_loader.format_docs, "question": RunnablePassthrough()}
    | prompt
    | ov_llm
    | StrOutputParser()
)
print("-------RAG Chain Answer for question---------")
for chunk in rag_chain.stream(args.q):
    print(chunk, end="", flush=True)

print("\n")
