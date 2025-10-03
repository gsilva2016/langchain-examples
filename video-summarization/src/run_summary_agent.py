import argparse
import asyncio
import time
from common.milvus.milvus_wrapper import MilvusManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from datetime import datetime
from common.summary_agent.agent import root_agent
import os


async def adk_runner(args): ##, last_processed_ts):
    milvus_manager = MilvusManager()    
    # print("last processed time:", last_processed_ts)
    filter_expr = '' ###f'metadata["event_creation_timestamp"] > "{last_processed_ts}"'
    collection_data = milvus_manager.query(
        collection_name=args.collection_name,
        filter=filter_expr,
        limit=1000,
        output_fields=["pk", "metadata", "vector"]
    )
    for item in collection_data:
        pk = item.get("pk")
        metadata = item.get("metadata", {})
        print(" pk:", pk)
        print("\n metadata:", metadata)
    
    # generate_end_of_day_report(collection_data, 'out.csv')
    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        return [], last_processed_ts  
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "report_agent_app"
    
    initial_state = {
        "collection_data": collection_data,
        "output_csv_path": "end_of_day_report.csv" 
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
        parts=[types.Part(text="use report_tool to generate end of day report")]
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
        # Assumes all timestamps are comparable and ISO
        latest_ts = max(item["metadata"]["event_creation_timestamp"] for item in collection_data)
        print("latest_ts", latest_ts)
        return collection_data, latest_ts
    else:
        return [], last_processed_ts


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    args = parser.parse_args()
    
    asyncio.run(adk_runner(args))
    

