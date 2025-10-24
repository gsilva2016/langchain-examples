from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
import os
import subprocess

def metadata_extractor(tool_context: ToolContext, collection_name: str) -> dict:
    milvus_manager = tool_context.state.get("milvus_manager")
    query_text = tool_context.state.get("query")
    
    collection_data = milvus_manager.query(
        collection_name=collection_name,
        filter='metadata["idle_status"] IS NOT NULL',
        limit=10,
        output_fields=["pk", "metadata", "vector"]
    )

    metadata_records = []
    for item in collection_data:
        metadata = item.get("metadata", {})
        metadata_records.append(metadata)

    return {
        "query": query_text,
        "collection": collection_name,
        "metadata": metadata_records
    }

def rag_retriever(tool_context: ToolContext) -> str:
    print("[rag_retriever] Launching RAG subprocess...")
    state = tool_context.state

    query_text = state.get("query", "")
    filter_expr = state.get("filter_expression", "")
    query_img = state.get("query_img", "")
    milvus_uri = state.get("MILVUS_HOST", "localhost")
    milvus_port = state.get("MILVUS_PORT", "19530")
    milvus_dbname = state.get("MILVUS_DBNAME", "default")
    collection_name = state.get("VIDEO_COLLECTION_NAME", "video_chunks")
    retrieve_top_k = str(state.get("RETRIEVE_TOP_K", 1))
    save_video_clip = str(state.get("SAVE_VIDEO_CLIP", False))
    video_clip_duration = str(state.get("VIDEO_CLIP_DURATION", 10))
    
    # Get absolute path to current file
    current_file = os.path.abspath(__file__)
    agents_folder = os.path.abspath(os.path.join(current_file, "../../.."))

    env = os.environ.copy()
    env["PYTHONPATH"] = agents_folder
    env["TOKENIZERS_PARALLELISM"] = "true"

    command = [
        "python", "src/rag.py",
        "--query_text", query_text,
        "--filter_expression", filter_expr,
        "--query_img", query_img,
        "--milvus_uri", milvus_uri,
        "--milvus_port", milvus_port,
        "--milvus_dbname", milvus_dbname,
        "--collection_name", collection_name,
        "--retrieve_top_k", retrieve_top_k,
        "--save_video_clip", save_video_clip,
        "--video_clip_duration", video_clip_duration
    ]

    result = subprocess.run(command, env=env, capture_output=True, text=True)
    print("[rag_retriever] Subprocess completed with code:", result.returncode)
    print("[rag_retriever] Output:\n", result.stdout)
    if result.stderr:
        print("[rag_retriever] Errors:\n", result.stderr)
    return result.stdout.strip() or result.stderr.strip()
