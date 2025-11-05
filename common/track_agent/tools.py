from datetime import datetime, timedelta
from google.adk.tools.tool_context import ToolContext

def price_alert_update(tool_context: ToolContext) -> dict:
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])

    available_agents_count = 0
    seen_agents = set()
    pks = []
    vectors = []
    metadatas = []

    threshold = 0.8  
    for item in collection_data:
        
        metadata = item.get("metadata", {})
        is_assigned = metadata['is_assigned']
        last_update_str = metadata['last_update']
        global_track_id = metadata['global_track_id']

        last_update_sec = None
        try:
            if last_update_str:
                last_update_sec = float(last_update_str)
        except Exception:
            continue
   
        if global_track_id and not is_assigned and last_update_sec:
            if global_track_id not in seen_agents:
                seen_agents.add(global_track_id)
                available_agents_count += 1
                
    for item in collection_data:
        pk = item.get("pk")
        vector = item.get("vector")
        if not pk or vector is None:
            continue
        
        metadata = item.get("metadata", {})
        is_assigned = metadata['is_assigned']
        last_update_str = metadata['last_update']
        deliveries_count = metadata['deliveries_count']
        event_creation_time_str= metadata['event_creation_timestamp']
        event_creation_time = datetime.strptime(event_creation_time_str, "%Y-%m-%d %H:%M:%S")
        event_creation_time_top_hour = event_creation_time.replace(minute=0, second=0, microsecond=0)
        event_creation_time_top_hour_str = event_creation_time_top_hour.strftime("%Y-%m-%d %H:%M:%S")
      
        metadata['available_agents'] = available_agents_count
        metadata['price_alert_time'] = event_creation_time_top_hour_str
        ratio = float(available_agents_count) / deliveries_count if deliveries_count > 0 else 1.0
        
        if ratio < threshold:
            price_summary_msg = (
                f"price alert at {event_creation_time}, agents={available_agents_count}, deliveries={deliveries_count}, ratio={ratio:.2f}"
            )
            price_alert_status= True
        else:
            price_summary_msg = "No alerts detected"
            price_alert_status= False

        metadata['price_alert_summary'] = price_summary_msg
        metadata['price_alert_status'] = price_alert_status
        
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
