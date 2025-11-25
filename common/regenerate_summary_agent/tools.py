from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
from typing import List
import re
from collections import OrderedDict

def extract_visible_ids(summary):
    # Check if summary says no visible individuals
    if re.search(r'No visible individuals in the video', summary, re.IGNORECASE):
        return []

    # Extract IDs after '- ' or after 'GID:'
    candidates = []
    # Pattern for '- ef9ba4' or similar
    candidates += re.findall(r'-\s*([a-zA-Z0-9]{6})', summary)
    # Pattern for '- GID: 000001'
    candidates += re.findall(r'GID:\s*([a-zA-Z0-9]{6})\b', summary)
    valid_ids = [cid for cid in candidates if cid.isdigit() or (re.search(r'[A-Za-z]', cid) and re.search(r'\d', cid))]
    
    # Filter: include if purely numeric OR alphanumeric with at least one digit

    # Remove duplicates while preserving order
    unique_ids = list(OrderedDict.fromkeys(valid_ids))
    return unique_ids



    
def process_summary(summary, start_time, end_time, logs):
    start_sec = float(start_time)
    end_sec = float(end_time)

    # Step 1: Track latest event per person within the timeframe
    latest_logs = {}
    for log in logs:
        log_meta = log.get("metadata", {})
        track_id = log_meta.get("global_track_id")
        last_update = log_meta.get("last_update")

        if not track_id or not last_update:
            continue

        last_update_val = int(float(last_update))

        # Only consider events within the timeframe
        if start_sec <= last_update_val <= end_sec:
            if track_id not in latest_logs or last_update_val > int(float(latest_logs[track_id]["metadata"]["last_update"])):
                latest_logs[track_id] = log

    # Step 2: Collect descriptions for detected IDs
    detected_info = [(track_id[:6], log["metadata"].get("description", "")) for track_id, log in latest_logs.items()]

    # Step 3: Compare with summary
    mentioned_ids = extract_visible_ids(summary)
    print("mentioned_ids: \n", mentioned_ids)

    #If no ids in summary, do nothing
    if len(mentioned_ids) == 0:
        return summary

    if not detected_info:
        return f"No one was detected between {start_sec} and {end_sec}."

    for gid, desc in detected_info:
        print("detected_info gid: \n", gid)
        if gid not in mentioned_ids:
            summary += f"\nGlobal ID {gid} was detected: {desc}"

    for mid in mentioned_ids:
        if mid not in [gid for gid, _ in detected_info]:
            summary += f"\nGlobal ID {mid} mentioned in summary but not detected in this time frame."

    return summary   

def enhance_summary(tool_context: ToolContext) -> dict:
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    collection_log_data = tool_context.state.get('collection_log_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])

    updated_summaries = []
    pks = []
    vectors = []
    metadatas = []

    # Loop through each video chunk in collection_data
    for item in collection_data:
        pk = item.get("pk")
        vector = item.get("vector")
        if not pk or vector is None:
            continue

        meta = item.get("metadata", {})
        summary = meta.get("summary", "")
        start_time = meta.get("start_time")
        end_time = meta.get("end_time")

        # Apply enhancement logic
        updated_summary = process_summary(summary, start_time, end_time, collection_log_data)

        # Update metadata with enhanced summary
        meta['updated_summary'] = updated_summary

        # Prepare for Milvus upsert
        pks.append(pk)
        vectors.append(vector)
        metadatas.append(meta)

        # Track results for reporting
        updated_summaries.append({
            "start_time": start_time,
            "end_time": end_time,
            "updated_summary": updated_summary
        })

    if not pks:
        print("No valid items found to upsert.")
        return {"status": "error", "message": "No valid items to upsert."}

    # Upsert updated summaries into Milvus
    try:
        result = milvus_manager.upsert_data(
            collection_name=collection_name,
            pks=pks,
            vectors=vectors,
            metadatas=metadatas
        )
        summary_msg = f"Agent updated DB successfully with {len(pks)} enhanced summaries."
        return {"status": "success", "message": summary_msg} ##, "details": updated_summaries}
    except Exception as ex:
        return {"status": "error", "message": f"Agent failed to update DB: {str(ex)}"}
    
# def enhance_summary(tool_context: ToolContext) -> dict:

    # collection_log_data = tool_context.state.get('collection_log_data', []) 
    # collection_data = tool_context.state.get('collection_data', [])
    # updated_summaries = []
    
    # # Process all summaries inside the function
    # for item in collection_data:
        # meta = item.get("metadata", {})
        # summary = meta.get("summary", "")
        # start_time = meta.get("start_time")
        # end_time = meta.get("end_time")
        
        # # Apply your enhancement logic here
        # updated_summary = process_summary(summary, start_time, end_time, collection_log_data)
        
        # updated_summaries.append({
            # "start_time": start_time,
            # "end_time": end_time,
            # "updated_summary": updated_summary
        # })
    
    # return updated_summaries

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

