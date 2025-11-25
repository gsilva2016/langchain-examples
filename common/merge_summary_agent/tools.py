from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
from typing import List

def iterative_summarization(summaries: List[str], group_size: int = 4) -> str:
    while len(summaries) > 1:
        new_summaries = []
        for i in range(0, len(summaries), group_size):
            chunk = summaries[i:i+group_size]
            # new_summary = summarize_texts(chunk)
            new_summaries.append(chunk)
        summaries = new_summaries
        group_size = 2  # After first pass, switch to summarizing pairs
    return summaries[0]
    
def summary_merger(tool_context: ToolContext) -> dict:
    """
    Updates idle status and summary for each global_track_id based on the latest last_update event.

    Args:
        tool_context: ADK-injected context object providing session state. Must include:
            - 'collection_name': the target collection name
            - 'collection_data': list of data entries with metadata containing 'global_track_id', 'first_detected', 'last_update', etc.
            - 'milvus_manager': Milvus manager instance used for data upsert
            - 'idle_threshold_seconds': threshold in seconds to consider an entity idle (default 900)

    Returns:
        dict: Status dictionary with keys:
            - 'status': 'success' or 'error'
            - 'message': additional info or error description
    """
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])
    
    
    batch = []
    final_summaries = []
    
    for item in collection_data:
        metadata = item.get("metadata", {})
        summary = metadata.get("summary")
    
        if summary:
            batch.append(summary)
    
        if len(batch) == 4:
            merged = " ".join(batch)
            refined = iterative_summarization(merged)
            final_summaries.append(refined)
            batch = []  # reset for next batch
    
    
    if batch:
        merged = " ".join(batch)
        refined = iterative_summarization(merged)
  
    if refined:
        print("final sumaary \n",refined)
        return "success"
    else:
        return "error"

