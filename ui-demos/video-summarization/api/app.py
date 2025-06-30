from merger.summary_merger import SummaryMerger
from datetime import datetime
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus.orm import utility
from pymilvus import db
from langchain_community.embeddings.openvino import OpenVINOEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import Collection, connections


class VideoChunk(BaseModel):
    chunk_id: str
    chunk_path: str
    chunk_summary: str
    start_time: str
    end_time: str

class IngestTxtRequest(BaseModel):
    data: List[VideoChunk]

class SummaryMergerRequest(BaseModel):
    """
    Pydantic model for the request body of the merge_summaries endpoint. It expects a dictionary of summaries.
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }
    """
    summaries: dict


class SummaryMergerResponse(BaseModel):
    """
    Pydantic model for the response body of the merge_summaries endpoint.
    It will return the overall summary and the anomaly score.
    """
    overall_summary: str
    anomaly_score: float


app = FastAPI()
summary_merger = SummaryMerger(device="GPU", max_new_tokens=275)

# Pieces pulled out from langchain-examples, not release ready code
txt_vectorstore = None
device = "CPU"
model_kwargs = {"device": device}
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}
ov_txt_embeddings = OpenVINOEmbeddings(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

# Connect to Milvus
db_name = "milvus_db"
milvus_uri = "localhost"
milvus_port = 19530

try:
    conn = connections.connect(host=milvus_uri, port=milvus_port)
    if db_name not in db.list_database():
        db.create_database(db_name)
    db.using_database(db_name)

    collections = utility.list_collections()
    for name in collections:
        # Collection(name).drop()
        print(f"Collection {name} exists.")

    # Create Milvus instance
    txt_vectorstore = Milvus(
        embedding_function=ov_txt_embeddings,
        collection_name="chunk_summaries",
        connection_args={"uri": f"http://{milvus_uri}:{milvus_port}", "db_name": db_name},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        consistency_level="Strong",
        drop_old=False,
    )

    print("Connected to Milvus DB successfully.")

except Exception as e:
    print(f"Error connecting to Milvus DB: {e}")
    sys.exit(1)

@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello from App."}


@app.post("/merge_summaries")
def merge_summaries(request: SummaryMergerRequest):
    """
    Endpoint for calling summary merger. Input should be in this format:
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }"""
    output = summary_merger.merge_summaries(request.summaries)
    return SummaryMergerResponse(**output)


@app.post("/embed_txt_and_store")
async def embed_txt_and_store(request: IngestTxtRequest):
    try:
        documents = [
            Document(
                page_content=item.chunk_summary,
                metadata={
                    "chunk_id": item.chunk_id,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "chunk_path": item.chunk_path,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            for item in request.data
        ]

        ids = [f"{doc.metadata['chunk_id']}" for doc in documents]

        txt_vectorstore.add_documents(documents=documents, ids=ids)

        return {"status": "success", "total_chunks": len(documents)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/archive_to_gcs")
async def archive_to_gcs(vector: List[str]):
    pass


@app.get("/query")
async def query(expr: str, collection_name: str = "chunk_summaries"):
    try:
        collection = Collection(collection_name)
        collection.load()

        results = collection.query(expr, collection_name=collection_name,
                                   output_fields=["chunk_id", "chunk_path", "start_time", "end_time"])
        print(f"{len(results)} vectors returned for query: {expr}")

        return {"status": "success", "chunks": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(query: str, top_k: int = 1):
    try:
        print(query)

        results = txt_vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=None,
            include=["metadata"],
        )

        return {
            "status": "success",
            "results": [
                {
                    "chunk_id": doc.metadata["chunk_id"],
                    "start_time": doc.metadata["start_time"],
                    "end_time": doc.metadata["end_time"],
                    "chunk_path": doc.metadata["chunk_path"],
                    "chunk_summary": doc.page_content
                }
                for doc in results
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
