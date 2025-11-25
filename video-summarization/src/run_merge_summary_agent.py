import argparse
import asyncio
import time
from common.milvus.milvus_wrapper import MilvusManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime, timedelta
from common.merge_summary_agent.agent import summarizer_agent
import os
from typing import List
import re
# def iterative_summarization(summaries: List[str], group_size: int = 4) -> str:
    # while len(summaries) > 1:
        # new_summaries = []
        # for i in range(0, len(summaries), group_size):
            # chunk = summaries[i:i+group_size]
            # # Replace with actual summarization logic
            # new_summaries.append(" ".join(chunk))  # Just merging for now
        # summaries = new_summaries
        # group_size = 2  # After first pass, switch to summarizing pairs
    # return summaries[0] if summaries else ""

# def summary_merger(collection_data) -> str:
    # batch = []
    # final_summaries = []

    # for item in collection_data:
        # metadata = item.get("metadata", {})
        # summary = metadata.get("summary")

        # if summary:
            # batch.append(summary)

        # # Process batch when it reaches 4 summaries
        # if len(batch) == 4:
            # refined = iterative_summarization(batch)
            # final_summaries.append(refined)
            # batch = []  # reset for next batch

    # # Process any remaining summaries (less than 4)
    # if batch:
        # refined = iterative_summarization(batch)
        # final_summaries.append(refined)

    # # Final check: return success if any refined summary is non-empty
    # if final_summaries and any(s.strip() for s in final_summaries):
        # print("Final summaries:\n", final_summaries)
        # return "success"
    # else:
        # return "error"


# async def adk_runner(args):
    # milvus_manager = MilvusManager()    
    
     # # Set initial timestamp to midnight of today
    # today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
    # # Calculate tomorrow's date at midnight (start of next day)
    # tomorrow = datetime.now().date() + timedelta(days=1)
    # tomorrow_midnight = datetime.combine(tomorrow, datetime.min.time()) 
    
    # filter_expr = f'metadata["mode"]=="text" AND metadata["db_entry_timestamp"] > "{today_midnight}" AND metadata["db_entry_timestamp"] < "{tomorrow_midnight}"'
  
    # collection_data = milvus_manager.query(
        # collection_name=args.collection_name,
        # filter=filter_expr,
        # limit=1000,
        # output_fields=["pk", "metadata", "vector"]
    # )
    # # for item in collection_data:
        # # metadata = item.get("metadata", {})
        # # print("\n metadata",metadata["summary"])
    # if not collection_data:
        # print("No data found in the collection. Skipping agent invocation.")
        # return [] 
    # result = summary_merger(collection_data) 
    # session_service = InMemorySessionService()
    # user_id = "user1"
    # session_id = "session1"
    # app_name = "idle_agent_app"
    # initial_state = {
        # "collection_name": collection_name,
        # "collection_data": collection_data,
        # "milvus_manager": milvus_manager,
        # "idle_threshold_seconds": interval_seconds,
    # }
    
    # # Create session asynchronously, with  initial state
    # session = await session_service.create_session(
        # app_name=app_name,
        # user_id=user_id,
        # session_id=session_id,
        # state=initial_state
    # )
    
    # start_time = time.monotonic()
    # runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)
            
   
    # user_input = types.Content(
        # role='user',
        # parts=[types.Part(text="use summary_merger_tool to generate a 15 minutes video summary based on short 30 seconds chunks summary")]
    # )
   
    
    # response_events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
    
    # async for event in response_events:
        # print("Agent output:", event.content.parts[0].text if event.content else None)
        # if event.is_final_response():
            # final_answer = event.content.parts[0].text if event.content else "No final response content"
            # print("Final agent response:", final_answer)
            # break
            
    # end_time = time.monotonic()
    # elapsed_seconds = end_time - start_time
    # minutes = int(elapsed_seconds // 60)
    # seconds = int(elapsed_seconds % 60)
    # print(f"Processing time: {minutes} minutes and {seconds} seconds")    
    # if collection_data:
        # latest_ts = (last_processed_dt + timedelta(minutes=14, seconds=59)) 
        # print("latest_ts", latest_ts)
        # return collection_data, latest_ts
    # else:
        # return [], last_processed_dt

 
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--milvus_uri", type=str, default="localhost")
    # parser.add_argument("--milvus_port", type=int, default=19530)
    # parser.add_argument("--milvus_dbname", type=str, default="default")
    # parser.add_argument("--collection_name", type=str, default="tracking_logs")
    # args = parser.parse_args()
    
  
    # asyncio.run(adk_runner(args))


# # Setup session and runner
# session_service = InMemorySessionService()
# runner = Runner(agent=manager_agent, app_name="video_summary_app", session_service=session_service)

# def make_user_message(summaries):
    # # Format to send as user message to manager agent
    # return types.Content(role="user", parts=[types.Part(text="\n".join(summaries))])

# def run_recursive_summarization(args):
    # user_id = "user1"
    # session_id = "session1"
    # session_service.create_session(app_name="video_summary_app", user_id=user_id, session_id=session_id)

    # message = make_user_message(initial_summaries)
    # events = runner.run_sync(user_id=user_id, session_id=session_id, new_message=message)

    # for event in events:
        # if event.content:
            # print(f"[{event.author}]: {event.content.parts[0].text}")

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--milvus_uri", type=str, default="localhost")
    # parser.add_argument("--milvus_port", type=int, default=19530)
    # parser.add_argument("--milvus_dbname", type=str, default="default")
    # parser.add_argument("--collection_name", type=str, default="tracking_logs")
    
    # run_recursive_summarization(args)


# from google.adk.agents import Agent
# from google.adk.tools import AgentTool
# from google.adk.sessions import InMemorySessionService
# from google.adk.runners import Runner
# from google.genai import types

# # Summarizer Agent
# summarizer_agent = Agent(
    # name="summarizer_agent",
    # model="gemini-2.0-flash",
    # description="Summarizes a list of video chunk summaries.",
    # instruction="Summarize the following text chunks clearly and concisely."
# )
# summarizer_tool = AgentTool(agent=summarizer_agent)

# # Manager Agent orchestrating recursive summarization using the summarizer tool
# manager_agent = Agent(
    # name="manager_agent",
    # model="gemini-2.0-flash",
    # description="Recursively summarizes batches of summaries until a final single summary is produced.",
    # instruction=(
        # "You receive one or more video chunk summaries. If multiple summaries are given, group them in batches of 4, "
        # "summarize each batch using the summarizer tool, and recursively summarize the intermediate results until only one summary remains."
    # ),
    # tools=[summarizer_tool]
# )


def query_summaries_from_db(collection_data, batch_size=4):
    intermediate_summaries = []
    batch = []
    for idx, item in enumerate(collection_data):
        metadata = item.get("metadata", {})
        # print("***\n", metadata)
        summary = metadata.get("updated_summary")
        # print("\n summary",summary)
        if summary:
            batch.append(summary)
        # When batch is full or last item reached, summarize current batch
        if len(batch) == batch_size or idx == len(collection_data) - 1:
            yield batch
            batch = []

def make_user_message(summaries):
    return types.Content(role="user", parts=[types.Part(text="\n".join(summaries))])
def clean_summary(text):
    if not text:
        return ""
    # Remove <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()    
async def run_recursive_summarization(args):
    milvus_manager = MilvusManager()    
    
    # Set initial timestamp to midnight of today
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
    tomorrow = datetime.now().date() + timedelta(days=1)
    tomorrow_midnight = datetime.combine(tomorrow, datetime.min.time()) 

    filter_expr = f'metadata["mode"]=="text" AND metadata["video_path"]=="Copy_of_D02_20250918150609_001_16mins.mp4" AND metadata["updated_summary"] IS NOT NULL' ## AND metadata["db_entry_timestamp"] > "2025-11-14 00:00:00" AND metadata["db_entry_timestamp"] < "2025-11-15 00:00:00"' # AND metadata["timestamp"] < "2025-11-07T10:00:00"'  # You can add timestamp filters if needed
   
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
    # filter_expr_track = "" ##f'metadata["event_creation_timestamp"] > "2025-11-14 00:00:00" AND metadata["event_creation_timestamp"] < "2025-11-15 00:00:00"'
    # print("Querying Milvus track_logs collection...")
    # collection_log_data = milvus_manager.query(
        # collection_name="tracking_logs",
        # filter=filter_expr_track,
        # limit=1000,
        # output_fields=["pk", "metadata", "vector"]
    # )
    
    # for item in collection_log_data:
        # metadata = item.get("metadata", {})
        # print("\n metadata track_logs \n ",metadata)
    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        return [] 
    
    # if not collection_log_data:
        # print("No data found in the log collection. Skipping agent invocation.")
        # return []     
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "video_summary_app"

    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=summarizer_agent, app_name=app_name, session_service=session_service)

    # Initial batch-wise summarization of DB summaries
    intermediate_summaries = []
    batch_count = 0
    for batch in query_summaries_from_db(collection_data, batch_size=4):
        batch_count += 1
        print(f"Processing batch {batch_count} with {len(batch)} summaries")
        input_msg = make_user_message(batch)
        events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
        async for event in events:
            if event.content:
                summary_text = clean_summary(event.content.parts[0].text)
                print(f"Intermediate summary {batch_count}:\n{summary_text}\n") ##[:300]}...")  
                intermediate_summaries.append(summary_text)

    print(f"Total intermediate summaries: {len(intermediate_summaries)}")

    # Recursive summarization until one final summary remains
    round_num = 1
    while len(intermediate_summaries) > 1:
        print(f"\n **** Starting recursive round {round_num} with {len(intermediate_summaries)} summaries ***")
        new_intermediates = []
        for i in range(0, len(intermediate_summaries), 4):
            batch = intermediate_summaries[i:i+4]
            print(f"  Summarizing batch {i//4 + 1} of size {len(batch)}")
            print(f"  batch summaries {batch} ")
            input_msg = make_user_message(batch)
            events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
            async for event in events:
                if event.content:
                    summary_text = clean_summary(event.content.parts[0].text)
                    print(f"  → New intermediate summary:\n{summary_text}\n")
                    new_intermediates.append(summary_text)
        intermediate_summaries = new_intermediates
        round_num += 1

    print("\n✅ Final summary:")
    print(intermediate_summaries[0] if intermediate_summaries else "No summaries found")
# async def run_recursive_summarization(args):
    # milvus_manager = MilvusManager()    
    
     # # Set initial timestamp to midnight of today
    # today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
    # # Calculate tomorrow's date at midnight (start of next day)
    # tomorrow = datetime.now().date() + timedelta(days=1)
    # tomorrow_midnight = datetime.combine(tomorrow, datetime.min.time()) 
    # filter_expr = f'metadata["mode"]=="text"' ## AND metadata["db_entry_timestamp"] > "{today_midnight}" AND metadata["db_entry_timestamp"] < "{tomorrow_midnight}"'
  
    # collection_data = milvus_manager.query(
        # collection_name=args.collection_name,
        # filter=filter_expr,
        # limit=1000,
        # output_fields=["pk", "metadata", "vector"]
    # )
    
    # if not collection_data:
        # print("No data found in the collection. Skipping agent invocation.")
        # return [] 
    
    # session_service = InMemorySessionService()
    
    # user_id = "user1"
    # session_id = "session1"
    # app_name = "video_summary_app"

    # session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    
    # start_time = time.monotonic()
    # runner = Runner(agent=summarizer_agent, app_name=app_name, session_service=session_service)
    # # Initial batch-wise summarization of DB summaries
    # intermediate_summaries = []
    # for batch in query_summaries_from_db(collection_data, batch_size=4):
        # input_msg = make_user_message(batch)
        # print("\n input_msg",input_msg)
        # events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
        # async for event in events:
            # if event.content:
                # intermediate_summaries.append(event.content.parts[0].text)
                # print(" intermediate summaries size", len(intermediate_summaries))
                # print(" intermediate summaries", intermediate_summaries)
    # # Recursive summarization on intermediate summaries until single summary remains
    # while len(intermediate_summaries) > 1:
        # new_intermediates = []
        # for i in range(0, len(intermediate_summaries), 4):
            # batch = intermediate_summaries[i:i+4]
            # input_msg = make_user_message(batch)
            # events = runner.run_async(user_id=user_id, session_id=session_id, new_message=input_msg)
            # async for event in events:
                # if event.content:
                    # new_intermediates.append(event.content.parts[0].text)
        # intermediate_summaries = new_intermediates

    # print("Final summary:", intermediate_summaries[0] if intermediate_summaries else "No summaries found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    
    args = parser.parse_args()
    asyncio.run(run_recursive_summarization(args))
