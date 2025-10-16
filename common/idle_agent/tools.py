from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
from datetime import datetime
from collections import defaultdict
from datetime import datetime

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
        if not global_track_id or not last_update_str:
            continue
        try:
            last_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        if (global_track_id not in grouped_latest) or (last_dt > grouped_latest[global_track_id]["last_dt"]):
            grouped_latest[global_track_id] = {"item": item, "last_dt": last_dt}

    # Step 2: Compute idle status and update metadata only for latest entries
    pks = []
    vectors = []
    metadatas = []

    for global_track_id, info in grouped_latest.items():
        item = info["item"]
        metadata = item.get("metadata", {})
        pk = item.get("pk")
        vector = item.get("vector")

        first_detected_str = metadata.get('first_detected')
        last_update_str = metadata.get('last_update')
        is_assigned = metadata.get('is_assigned', False)

        if not first_detected_str or not last_update_str:
            continue

        first_dt = datetime.strptime(first_detected_str, "%Y-%m-%d %H:%M:%S")
        last_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        idle = (last_dt - first_dt).total_seconds() >= idle_threshold_seconds and not is_assigned

        metadata['idle_status'] = idle
        seen_in_videos = ', '.join(metadata.get('seen_in', []))
        idle_text = "idle" if idle else "not idle"
        metadata['summary'] = (
            f"{global_track_id} was first seen at {first_detected_str}, last seen at {last_update_str} and event created at {metadata.get('event_creation_timestamp')}, "
            f"in videos ({seen_in_videos}), and is {idle_text}."
        )

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