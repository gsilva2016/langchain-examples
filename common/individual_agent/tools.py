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

        # Convert timestamp to start-of-the-hour (e.g., 10:34 → 10:00))
        event_dt = datetime.strptime(event_ts, "%Y-%m-%dT%H:%M:%S")

        hour_start = event_dt.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        hour_window = f"{hour_start.strftime('%H:%M')}-{hour_end.strftime('%H:%M')}"
        first_detected = metadata.get("first_detected")
        last_update = metadata.get("last_update")
        key = (global_track_id, hour_start)
        if key not in hourly_reports:
            hourly_reports[key] = {
                "global_track_id": global_track_id,
                "hour_window": hour_window,
                "summary_count": 1,
                "idle_status": idle_status,
                "first_seen": first_detected,
                "last_seen": last_update
            }
        else:
            hourly_reports[key]["summary_count"] += 1
            # Keep earliest first_seen
            prev_first = hourly_reports[key]["first_seen"]
            if first_detected and (not prev_first or first_detected < prev_first):
                hourly_reports[key]["first_seen"] = first_detected
            # Keep latest last_seen
            prev_last = hourly_reports[key]["last_seen"]
            if last_update and (not prev_last or last_update > prev_last):
                hourly_reports[key]["last_seen"] = last_update

    vectors = []
    metadatas = []

    for (track_id, hour_dt), report in hourly_reports.items():
        summary = (
            f"Individual {track_id} detected {report['summary_count']} times in {report['hour_window']}, "
            f"was {report['idle_status']}, first seen at {report['first_seen']}, "
            f"and last seen at {report['last_seen']}."
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
