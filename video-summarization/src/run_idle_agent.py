import argparse
import asyncio
import time
from common.milvus.milvus_wrapper import MilvusManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta
from common.idle_agent.agent import root_agent
import os


async def adk_runner(args, last_processed_dt, interval_seconds):
    milvus_manager = MilvusManager()    
    print("last processed time:", last_processed_dt)
    fifteen_minutes_later = last_processed_dt + timedelta(seconds=interval_seconds)
    print("fifteen_minutes_later:", fifteen_minutes_later)
    filter_expr = f'metadata["event_creation_timestamp"] > "{last_processed_dt}" AND metadata["event_creation_timestamp"] < "{fifteen_minutes_later}"'
    
    collection_data = milvus_manager.query(
        collection_name=args.collection_name,
        filter=filter_expr,
        limit=1000,
        output_fields=["pk", "metadata", "vector"]
    )
    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        last_processed_dt = (last_processed_dt + timedelta(minutes=14, seconds=59))
        return [], last_processed_dt        
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "idle_agent_app"
    initial_state = {
        "collection_name": "tracking_logs",
        "collection_data": collection_data,
        "milvus_manager": milvus_manager,
        "idle_threshold_seconds": interval_seconds,
    }
    
    # Create session asynchronously, with  initial state
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )
    
    start_time = time.monotonic()
    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)
            
   
    user_input = types.Content(
        role='user',
        parts=[types.Part(text="use idle_status_tool to store idle_status and summary in Milvus database")]
    )
   
    
    response_events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
    
    async for event in response_events:
        print("Agent output:", event.content.parts[0].text if event.content else None)
        if event.is_final_response():
            final_answer = event.content.parts[0].text if event.content else "No final response content"
            print("Final agent response:", final_answer)
            break
            
    end_time = time.monotonic()
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    print(f"Processing time: {minutes} minutes and {seconds} seconds")    
    if collection_data:
        latest_ts = (last_processed_dt + timedelta(minutes=14, seconds=59)) 
        print("latest_ts", latest_ts)
        return collection_data, latest_ts
    else:
        return [], last_processed_dt

async def periodic_runner(args, interval_seconds):
    # Set initial timestamp to midnight of today
    last_processed_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate tomorrow's date at midnight (start of next day)
    tomorrow = datetime.now().date() + timedelta(days=1)
    tomorrow_midnight = datetime.combine(tomorrow, datetime.min.time())
  
    while True:
        new_data, last_processed_dt = await adk_runner(args, last_processed_dt,interval_seconds)
        if not new_data and last_processed_dt >= tomorrow_midnight:
            print("No new data found, exiting loop.")
            break  # exit while True loop
        await asyncio.sleep(3) ##interval_seconds)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    args = parser.parse_args()
    
    load_dotenv()
    interval_minutes = float(os.getenv("RUNNER_INTERVAL_MINUTES", 15))  # Default is 15 min
    interval_seconds = int(interval_minutes * 60)

    asyncio.run(periodic_runner(args, interval_seconds))

