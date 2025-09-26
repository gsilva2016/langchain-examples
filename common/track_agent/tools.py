from datetime import datetime
import random

def surge_alert_update(tool_context) -> dict:
    collection_name = tool_context.state.get('collection_name', [])
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager', [])

    now = datetime.now()
    surge_time = now.replace(minute=0, second=0, microsecond=0)

    available_agents_count = 0
    pks = []
    vectors = []
    metadatas = []

    deliveries_count = random.randint(0, 100)
    threshold = 0.8  
    surge_time_str = surge_time.isoformat(sep=' ', timespec='seconds')
    for item in collection_data:
        pk = item.get("pk")
        vector = item.get("vector")
        if not pk or vector is None:
            continue
        
        metadata = item.get("metadata", {})
        is_assigned = metadata.get('is_assigned', False)
        last_update_str = metadata.get('last_update')
        last_update = None
        try:
            if last_update_str:
                last_update = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        
        if not is_assigned and last_update and last_update > surge_time:
            available_agents_count += 1
            
        metadata['surge_alert_time'] = surge_time_str
        metadata['available_agents'] = available_agents_count
        metadata['deliveries_count'] = deliveries_count
        ratio = float(available_agents_count) / deliveries_count if deliveries_count > 0 else 1.0
        
        if ratio < threshold:
            surge_summary_msg = (
                f"Surge alert at {surge_time_str}, agents={available_agents_count}, deliveries={deliveries_count}, ratio={ratio:.2f}"
            )
            surge_alert_status= True
        else:
            surge_summary_msg = "No alerts detected"
            surge_alert_status= False

        metadata['surge_summary'] = surge_summary_msg
        metadata['surge_alert'] = surge_alert_status
        
        pks.append(pk)
        vectors.append(vector)
        metadatas.append(metadata)

    try:
        milvus_manager.upsert_data(
            collection_name=collection_name,
            pks=pks,
            vectors=vectors,
            metadatas=metadatas
        )
        return f"Surge alert saved with agents={available_agents_count}, deliveries={deliveries_count}, time={surge_time_str}"
    except Exception as ex:
        return f"Failed to update surge alert: {str(ex)}"
