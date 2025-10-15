from datetime import datetime
from google.adk.tools.tool_context import ToolContext
from collections import defaultdict
from datetime import datetime

def update_idle_status(tool_context: ToolContext) -> dict:
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])
    idle_threshold_seconds = tool_context.state.get('idle_threshold_seconds', 900) 

    # Step 1: Group all entries by global_track_id
    grouped_entries = defaultdict(list)
    for item in collection_data:
        metadata = item.get("metadata", {})
        global_track_id = metadata.get("global_track_id")
        grouped_entries[global_track_id].append(item)

    # Step 2: For each person, select entry with maximum (last_update - first_detected)
    selected_items = []
    for global_track_id, items in grouped_entries.items():
        max_item = None
        max_interval = -1
        for item in items:
            metadata = item.get("metadata", {})
            first_detected_str = metadata.get('first_detected')
            last_update_str = metadata.get('last_update')
            if first_detected_str and last_update_str:
                first_dt = datetime.strptime(first_detected_str, "%Y-%m-%d %H:%M:%S")
                last_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
                interval = (last_dt - first_dt).total_seconds()
                if interval > max_interval:
                    max_interval = interval
                    max_item = item
        if max_item:
            selected_items.append(max_item)

    # Step 3: Prepare data for upsert with idle status/summary
    pks = []
    vectors = []
    metadatas = []
    for item in selected_items:
        pk = item.get("pk")
        vector = item.get("vector")
        metadata = item.get("metadata", {})
        first_detected_str = metadata['first_detected']
        last_update_str = metadata['last_update']
        is_assigned = metadata['is_assigned']
        first_dt = datetime.strptime(first_detected_str, "%Y-%m-%d %H:%M:%S")
        last_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        idle = (last_dt - first_dt).total_seconds() >= idle_threshold_seconds and not is_assigned
        metadata['idle_status'] = idle
        seen_in_videos = ', '.join(metadata.get('seen_in', []))
        idle_text = "idle" if idle else "not idle"
        metadata['summary'] = (
            f"{metadata.get('global_track_id')} was first seen at {first_detected_str}, "
            f"last seen at {last_update_str}, in videos ({seen_in_videos}), and is {idle_text}."
        )
        pks.append(pk)
        vectors.append(vector)
        metadatas.append(metadata)

    if not pks:
        print("No valid items found to upsert.")
        return {"status": "error", "message": "No valid items to upsert."}

    try:
        result = milvus_manager.upsert_data(collection_name=collection_name,
                                            pks=pks,
                                            vectors=vectors,
                                            metadatas=metadatas)
        summary = f"Agent updated DB successfully with {len(pks)} entries."
        return summary
    except Exception as ex:
        return f"Agent failed to update DB: {str(ex)}"