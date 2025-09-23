from datetime import datetime
from google.adk.tools.tool_context import ToolContext

def update_idle_status(tool_context: ToolContext) -> dict:
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])
    pks = []
    vectors = []
    metadatas = []
    idle_threshold_seconds= 30
    for item in collection_data:
        pk = item.get("pk")
        if not pk:
            continue
        
        vector = item.get("vector")
        if vector is None:
            continue
        
        metadata = item.get("metadata", {})
        first_detected_str = metadata['first_detected']
        last_update_str = metadata['last_update']
        is_assigned = metadata['is_assigned']
        if first_detected_str and last_update_str:
            first_dt = datetime.strptime(first_detected_str, "%Y-%m-%d %H:%M:%S")
            last_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
            idle = (last_dt - first_dt).total_seconds() >= idle_threshold_seconds and not is_assigned
            metadata['idle_status'] = idle
            seen_in_videos = ', '.join(metadata.get('seen_in', []))
            idle_text = "idle" if idle else "not idle"
            metadata['summary'] = (
                f"{metadata.get('global_track_id')} was first seen at {first_detected_str}, "
                f"last seen at {last_update_str}, in videos ({seen_in_videos}), and is currently {idle_text}."
            )
        else:
            metadata['idle_status'] = False
            metadata['summary'] = "Missing required timing information."
        #Debug prints
        # print(f"Processing pk={pk}, vector length={len(vector)}, metadata keys={list(metadata.keys())}")

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
