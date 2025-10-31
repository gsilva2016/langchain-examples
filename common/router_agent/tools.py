from datetime import datetime, timedelta, time
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
from dateparser.search import search_dates
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

def update_filter_with_time(query_text, current_filter):
    now = datetime.now()
    results = search_dates(query_text)
    if not results:
        return current_filter

    matched_text, dt = results[0]
      
    if "yesterday" in query_text.lower():
        start_time = datetime.combine(dt.date(), time.min)
        end_time = datetime.combine(dt.date(), time.max)
    elif any(kw in query_text.lower() for kw in ["month ago", "months ago", "day ago", "days ago"]):
        start_time = datetime.combine(dt.date(), time.min)
        end_time = datetime.combine(dt.date(), time.max)
    elif any(kw in query_text.lower() for kw in ["hour ago", "minute ago", "hours ago", "minutes ago"]):
        start_time = dt.replace(second=0, microsecond=0)
        end_time = start_time + timedelta(minutes=1, microseconds=-1)
    # Handle "today" as start and end of day
    elif "today" in query_text.lower():
        if any(t in query_text.lower() for t in ["am", "pm", ":"]):  # Specific time given
            start_time = dt
            end_time = dt + timedelta(seconds=1)  # narrow window of 1 second
        else:
            # No specific time, full day range
            start_time = datetime.combine(dt.date(), time.min)
            end_time = datetime.combine(dt.date(), time.max)
    elif any(kw in query_text.lower() for kw in ["last", "past", "in the past"]):
        if dt > now:
            duration = dt - now
        else:
            duration = now - dt
        start_time = now - duration
        end_time = now
    else:
        if dt.date() > now.date():
            dt = datetime.combine(now.date(), dt.time())
        start_time = dt
        end_time = dt

    time_filter = f'(metadata["timestamp"] >= "{start_time.isoformat()}") and (metadata["timestamp"] <= "{end_time.isoformat()}")'

    if current_filter:
        return f'({current_filter}) and ({time_filter})'
    else:
        return time_filter
def rag_retriever(tool_context: ToolContext) -> str:
    print("[rag_retriever] Launching RAG subprocess...")
    state = tool_context.state

    query_text = state.get("query", "describe the scene")
    filter_expr = state.get("filter_expression", 'metadata["mode"]=="text"')
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
    # Update filter expression based on detected time keywords
    updated_filter_expr = update_filter_with_time(query_text, filter_expr)
    
    command = [
        "python", "src/rag.py",
        "--query_text", query_text,
        "--filter_expression", updated_filter_expr,
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
