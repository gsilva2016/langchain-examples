import argparse
import asyncio
import time
from common.milvus.milvus_wrapper import MilvusManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta
from common.enhance_summary_agent.agent import enhance_summary_agent
import os
from typing import List
import re
from collections import OrderedDict



def query_summaries_from_db(collection_data, batch_size=4):
    intermediate_summaries = []
    batch = []
    for idx, item in enumerate(collection_data):
        metadata = item.get("metadata", {})
        print("***\n", metadata)
        summary = metadata.get("summary")
        # print("\n summary",summary)
        if summary:
            batch.append(summary)
        # When batch is full or last item reached, summarize current batch
        if len(batch) == batch_size or idx == len(collection_data) - 1:
            yield batch
            batch = []

def make_user_message(summaries):
    return types.Content(role="user", parts=[types.Part(text="\n".join(summaries))])
    
# async def run_recursive_summarization(args):
async def run_enhance_summarization(args):
    milvus_manager = MilvusManager()    
    
    # Set initial timestamp to midnight of today
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
    tomorrow = datetime.now().date() + timedelta(days=1)
    tomorrow_midnight = datetime.combine(tomorrow, datetime.min.time()) 

    filter_expr = f'metadata["mode"]=="text" AND metadata["video_path"]=="Copy_of_D02_20250918150609_001_16mins.mp4"' ## AND metadata["db_entry_timestamp"] > "2025-11-14 00:00:00" AND metadata["db_entry_timestamp"] < "2025-11-15 00:00:00"' # AND metadata["timestamp"] < "2025-11-07T10:00:00"'  # You can add timestamp filters if needed
   
    print("Querying Milvus collection...")
    collection_data = milvus_manager.query(
        collection_name=args.collection_name,
        filter=filter_expr,
        limit=1000,
        output_fields=["pk", "metadata", "vector"]
    )

    print(f"Total items retrieved from DB: {len(collection_data)}")

    # for item in collection_data:
        # metadata = item.get("metadata", {})
        # print("metadata \n",metadata)
    
    filter_expr_track = "" #f'metadata["event_creation_timestamp"] > "2025-11-14 00:00:00" AND metadata["event_creation_timestamp"] < "2025-11-15 00:00:00"'
    print("Querying Milvus track_logs collection...")
    collection_log_data = milvus_manager.query(
        collection_name="tracking_logs",
        filter=filter_expr_track,
        limit=1000,
        output_fields=["pk", "metadata", "vector"]
    )
    
    # for item in collection_log_data:
        # metadata = item.get("metadata", {})
        # print("\n metadata track_logs \n ",metadata)
    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        return [] 
    
    if not collection_log_data:
        print("No data found in the log collection. Skipping agent invocation.")
        return []     

   
   # # Example usage:
   # results = enhance_summary(collection_data, collection_log_data)

   ## Print results
   #for us in results:
   #     print(f"Time Frame: {us['start_time']} - {us['end_time']}\nUpdated Summary:\n{us['updated_summary']}\n")

        
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "enhance_summary_app"
    
    initial_state = {
        "collection_name": args.collection_name,
        "collection_log_data": collection_log_data,
        "collection_data": collection_data,
        "milvus_manager": milvus_manager,
    }

    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id, state=initial_state)
  
    runner = Runner(agent=enhance_summary_agent, app_name=app_name, session_service=session_service)
 
    user_input = types.Content(
        role='user',
        parts=[types.Part(text="use enhance_summary_tool to enhance summaries")]
    )
    
    response_events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
    
    async for event in response_events:
        if event.is_final_response():
            final_answer = event.content.parts[0].text if event.content else "No final response content"
            # final_answer = session.state.get("updated_summaries", "No tool output found")
            print("Final agent response:", final_answer)
            break


    # async for event in response_events:
        # print("Agent output:", event.content.parts[0].text if event.content else None)
        # if event.is_final_response():
            # final_answer = event.content.parts[0].text if event.content else "No final response content"
            # print("Final agent response:", final_answer)
            # break
    # # Initial batch-wise summarization of DB summaries
    # intermediate_summaries = []
    # batch_count = 0
    # for batch in query_summaries_from_db(collection_data, batch_size=4):
    #     batch_count += 1
    #     print(f"Processing batch {batch_count} with {len(batch)} summaries")
    #     input_msg = make_user_message(batch)
    #     events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
    #     async for event in events:
    #         if event.content:
    #             summary_text = event.content.parts[0].text
    #             print(f"Intermediate summary {batch_count}:\n{summary_text}\n") ##[:300]}...")  # Preview first 100 chars
    #             intermediate_summaries.append(summary_text)

    # print(f"Total intermediate summaries: {len(intermediate_summaries)}")

    # # Recursive summarization until one final summary remains
    # round_num = 1
    # while len(intermediate_summaries) > 1:
    #     print(f"\nStarting recursive round {round_num} with {len(intermediate_summaries)} summaries")
    #     new_intermediates = []
    #     for i in range(0, len(intermediate_summaries), 4):
    #         batch = intermediate_summaries[i:i+4]
    #         print(f"  Summarizing batch {i//4 + 1} of size {len(batch)}")
    #         input_msg = make_user_message(batch)
    #         events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
    #         async for event in events:
    #             if event.content:
    #                 summary_text = event.content.parts[0].text
    #                 print(f"  → New intermediate summary:\n{summary_text}\n") #[:300]}...")
    #                 new_intermediates.append(summary_text)
    #     intermediate_summaries = new_intermediates
    #     round_num += 1

    # print("\n✅ Final summary:")
    # print(intermediate_summaries[0] if intermediate_summaries else "No summaries found")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    
    args = parser.parse_args()
    asyncio.run(run_enhance_summarization(args))
