from datetime import datetime, timedelta
from google.adk.tools.tool_context import ToolContext
import numpy as np

def individual_report_generator(tool_context: ToolContext) -> dict:
    collection_data = tool_context.state.get('collection_data', [])
    milvus_manager = tool_context.state.get('milvus_manager')
    report_collection_name =tool_context.state.get('collection_name')
    hourly_reports = {}

    for item in collection_data:
        metadata = item.get("metadata", {})
        global_track_id = metadata.get("global_track_id")
        event_ts = metadata.get("event_creation_timestamp")
        if metadata.get("idle_status") == True:
            idle_status = 'idle'
        else:
            idle_status = 'not idle'          
        if not global_track_id or not event_ts:
            continue

        # Convert timestamp to start-of-the-hour (e.g., 10:34 → 10:00)
        # event_dt = datetime.fromtimestamp(event_ts)
        event_dt = datetime.strptime(event_ts, "%Y-%m-%dT%H:%M:%S")

        hour_start = event_dt.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        hour_window = f"{hour_start.strftime('%H:%M')}-{hour_end.strftime('%H:%M')}"
        # print( f' metadata: {metadata} \n')
        key = (global_track_id, hour_start)
        if key not in hourly_reports:
            hourly_reports[key] = {
                "global_track_id": global_track_id,
                "hour_window": hour_window,
                "summary_count": 1,
                "idle_status": idle_status
            }
        else:
            hourly_reports[key]["summary_count"] += 1

    vectors = []
    metadatas = []

    for (track_id, hour_dt), report in hourly_reports.items():
        summary = (
            f"Individual {track_id} detected {report['summary_count']} times in {report['hour_window']} and was {report['idle_status']}."
        )
        metadatas.append({
            "global_track_id": track_id,
            "hour_window": report["hour_window"],
            "summary": summary,
        })
        vectors.append(np.random.rand(256).tolist())

    try:
        milvus_manager.insert_data(
            collection_name=report_collection_name,
            vectors=vectors,
            metadatas=metadatas
        )
        return {"status": "success", "message": "Successfully inserted hourly individual reports."}
    except Exception as ex:
        return {"status": "error", "message": f"Failed to update hourly individual reports: {str(ex)}"}
