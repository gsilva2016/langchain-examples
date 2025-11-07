from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict

def format_seconds(seconds_float):
    total_seconds = int(seconds_float)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 or (hours == 0 and minutes == 0):
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return " and ".join(parts)
    
def update_idle_status(tool_context: ToolContext) -> dict:
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
    idle_threshold_seconds = tool_context.state.get('idle_threshold_seconds', 900)    

    # Step 1: For each global_track_id, retain only the entry with the maximum (most recent) last_update
    grouped_latest = {}
    for item in collection_data:
        metadata = item.get("metadata", {})
        global_track_id = metadata.get("global_track_id")
        last_update_str = metadata.get('last_update')
        event_type = metadata.get('event_type')
        if not global_track_id or not last_update_str:
            continue
        last_update_float = float(last_update_str)
        if (global_track_id not in grouped_latest) or (last_update_float > float(grouped_latest[global_track_id]["last_dt"])):
            grouped_latest[global_track_id] = {"item": item, "last_dt": last_update_str}

    # Step 2: Compute idle status and update metadata only for latest entries
    pks = []
    vectors = []
    metadatas = []
    latest_updates = {}

    for global_track_id, info in grouped_latest.items():
        item = info["item"]
        metadata = item.get("metadata", {})
        pk = item.get("pk")
        vector = item.get("vector")

        first_detected_str = metadata.get('first_detected')
        last_update_str = metadata.get('last_update')
        is_assigned = metadata.get('is_assigned', False)
        event_type = metadata.get('event_type')

        if not first_detected_str or not last_update_str:
            continue

        last_update = float(last_update_str)
        first_detected =float(first_detected_str)
        idle = (last_update - first_detected) >= idle_threshold_seconds and is_assigned

        metadata['idle_status'] = idle
        seen_in_videos = ', '.join(metadata.get('seen_in', []))
        idle_text = "idle" if idle else "not idle"
        
    
        # Update latest_updates dictionary
        latest_updates[global_track_id] = last_update
        
        seen_in_videos = ', '.join(metadata.get('seen_in', []))
        idle_text = "idle" if idle else "not idle"
    
        first_seen_formatted = format_seconds(first_detected)
        last_seen_formatted = format_seconds(last_update)
        entered_text = f"first entered at {first_seen_formatted}"
        exited_text = f"exited at {last_seen_formatted}" if event_type == "exit" else ""
    
        summary_parts = [
            f"{global_track_id} was {entered_text}",
            f"{exited_text}" if exited_text else "",
            f"event created at {metadata.get('event_creation_timestamp')}",
            f"in videos ({seen_in_videos})",
            f"and he is {idle_text}."
        ]

    
        metadata['summary'] = ", ".join(part for part in summary_parts if part)


        pks.append(pk)
        vectors.append(vector)
        metadatas.append(metadata)

    if not pks:
        return {"status": "error", "message": "No valid items with proper timestamps to upsert."}

    try:
        milvus_manager.upsert_data(
            collection_name=collection_name,
            pks=pks,
            vectors=vectors,
            metadatas=metadatas
        )
        return {
            "status": "success",
            "message": f"Updated {len(pks)} individuals based on their most recent last_update."
        }
    except Exception as ex:
        return {"status": "error", "message": f"Failed to update data: {str(ex)}"}
