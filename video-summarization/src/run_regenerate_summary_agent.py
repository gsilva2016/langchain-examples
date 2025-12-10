import argparse
import asyncio
import time
from common.milvus.milvus_wrapper import MilvusManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta
from common.regenerate_summary_agent.agent import regenerate_summary_agent
import os
import re

# Helper: Convert seconds to mm:ss
def format_time(seconds):
    try:
        sec = float(seconds)
        mins = int(sec // 60)
        secs = int(sec % 60)
        return f"{mins:02d}:{secs:02d}"
    except:
        return f"{seconds}s"

def clean_summary(text):
    if not text:
        return ""
    # Remove <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

async def run_enhance_summarization(args):
    milvus_manager = MilvusManager()    
  

    filter_expr = f'metadata["mode"]=="text"'
   
    print("Querying Milvus collection...")
    collection_data = milvus_manager.query(
        collection_name=args.collection_name,
        filter=filter_expr,
        limit=50,
        output_fields=["pk", "metadata", "vector"]
    )

    print(f"Total items retrieved from DB: {len(collection_data)}")

    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        return [] 
    

    # Prepare runner and session
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "enhance_summary_app"
    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id, state={})
    runner = Runner(agent=regenerate_summary_agent, app_name=app_name, session_service=session_service)

    updated_summaries = []
    pks, vectors, metadatas = [], [], []

    for item in collection_data:
        pk = item.get("pk")
        vector = item.get("vector")
        meta = item.get("metadata", {})
        original_summary = meta.get("summary", "")
        start_time = meta.get("start_time")
        end_time = meta.get("end_time")
        
        # Query Milvus for logs within this timeframe
        filter_expr_track = f'metadata["last_update"] >= {start_time} AND metadata["last_update"] <= {end_time}'
        
        collection_log_data = milvus_manager.query(
            collection_name="tracking_logs",
            filter=filter_expr_track,
            limit=20,
            output_fields=["pk", "metadata", "vector"])
     
        print(f"Total items retrieved from DB: {len(collection_log_data)} between {start_time} and {end_time} ")
     
        if not collection_log_data:
            updated_summary = f"No events detected between {start_time} and {end_time}."
            print(f"Time Frame: {start_time} - {end_time}\nUpdated Summary:\n{updated_summary}\n")
            print(f"No logs found for timeframe {start_time} - {end_time}. Skipping Agent call..")
            continue
        
        # Collect detected events for this timeframe
        detected_events = []
        for log in collection_log_data:
            log_meta = log.get("metadata", {})
            track_id = log_meta.get("global_track_id")
            if not track_id:
                continue

            gid = track_id[:6]
            event_type = log_meta.get("event_type", "detected")
            first_detected = log_meta.get("first_detected", "")
            camera = log_meta.get("last_camera", "")
            last_update = log_meta.get("last_update")

            if event_type == "exit":
                desc = f"Global ID {gid} exited at {format_time(last_update)} after entering at {format_time(first_detected)}."
            elif event_type == "entry":
                desc = f"Global ID {gid} entered at {format_time(first_detected)} and was last seen on camera {camera}."
            else:
                desc = f"Global ID {gid} was detected at {format_time(first_detected)} and last seen at {format_time(last_update)}."
            detected_events.append(desc)
      
        prompt = f"""
        You are rewriting a video summary using the original summary and detection metadata.
        
        Original summary:
        {original_summary}
        
        Timeframe: {start_time} - {end_time}
        
        Detected events:
        {chr(10).join(detected_events) if detected_events else "No detections in this timeframe."}
        
        Rewrite the summary with these rules:
        1.Preserve Original Times: Do not modify any detected entry or exit times; use them exactly as recorded.
        2.Include All Individuals: Mention every detected Global ID and their consolidated presence interval.
        3.Merge Multiple Detections: If multiple detections exist for the same Global ID, summarize their presence as a continuous range (e.g., “remained from 00:46 to 00:56”).
        4.If no detections: explicitly state "No one was detected between {start_time} and {end_time}."
        5.Integrate Details Naturally: Combine detection details into a narrative paragraph, not a bullet list.
        6.Mention Global IDs Clearly: Always include the Global ID in the summary.
        7.Briefly Describe New Individuals: When introducing a new Global ID, add a short phrase (e.g., “another person GID 0b1d2a entered at 00:54 and exited at 00:56”).
        8.Ensure Cohesion: Do not simply append detection logs; produce a single, well-structured summary.
        9.Camera Info (Optional): If camera details are available, integrate them naturally (e.g., “as seen on camera D02_20250918150609_001.mp4”).
        
        Return only the rewritten summary.
        """
       
        user_input = types.Content(role='user', parts=[types.Part(text=prompt)])
        response_events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
        
        updated_summary = None
        async for event in response_events:
            if event.is_final_response():
                updated_summary = clean_summary(event.content.parts[0].text if event.content else original_summary)

        # Update metadata
        meta['updated_summary'] = updated_summary or original_summary
        pks.append(pk)
        vectors.append(vector)
        metadatas.append(meta)

        updated_summaries.append({
            "start_time": start_time,
            "end_time": end_time,
            "updated_summary": updated_summary
        })

    # Upsert updated summaries into Milvus
    milvus_manager.upsert_data(args.collection_name, pks, vectors, metadatas)
    print(f"Updated {len(updated_summaries)} summaries successfully.")
    for us in updated_summaries:
        print(f"Time Frame: {us['start_time']} - {us['end_time']}\nUpdated Summary:\n{us['updated_summary']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    
    args = parser.parse_args()
    asyncio.run(run_enhance_summarization(args))
