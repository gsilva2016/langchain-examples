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


def query_summaries_from_db(collection_data, batch_size=5):
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
    prompt = f"""
    Task:
    You are responsible for summarizing these batches of video summaries. Each batch contains up to 5 summaries.
    
    Batches of summaries:
    {' '.join(summaries)}
    
    Your goal is to generate a unified summary that preserves all structured details while synthesizing information clearly.
    Context:
    The scene takes place in a warehouse or store environment.
    Rules:
    1.Preserve ALL Individuals (GIDs):
    Every Global ID (GID) mentioned in the input must appear in the final summary.
    Create a master list of all GIDs before merging. 
    2.Consolidate Time Intervals and Actions:
    For each GID, merge all timeframes from different summaries into a single consolidated range (or list if non-contiguous).
    List all distinct timeframes under one entry if they are non-contiguous.
    Include every observed action: entering, exiting, stationary, interacting with objects or people, bending, organizing, using a phone, etc.
    If a GID appears multiple times, combine details chronologically without losing unique actions.
    3.Preserve Context:
    Retain location references (e.g., “near entrance,” “storage area,” “bench”).
    Include object interactions (e.g., “picked up green bag,” “handled boxes,” “used phone”). 
    4.Highlight Key Behaviors and Transitions:
    Summarize interactions between individuals and notable movements (e.g., “GID X moved from entrance to storage area and interacted with GID Y”).
    Note any gaps in detection or inactivity periods.
    5.Avoid Redundancy:
    Do not copy individual summaries verbatim.
    Create one unified narrative that is clear, structured and somprehensive.
    6.Ensure Completeness:
    Before finalizing, cross-check the output against the master GID list to confirm all GIDs and their actions are included.
    Ensure no GID, timeframe or action is ommitted.
    7.Output Structure (mandatory at every level):
    **Overall Summary:** High-level description of the scene.
    **Detailed GID Actions:** Numbered list of all GIDs with consolidated timeframes and actions.
    **Key Transitions and Observations:** Interactions, movement patterns, anomalies.
    **Suspicious Activity Note:** Explicitly state if none detected.
    **Master GID List:** All unique GIDs included in this summary.
    8.Recursive Consistency:
    Apply these rules at every level of merging.
    Do NOT compress or omit structured details in higher-level summaries.
    If token limits require shortening, prioritize keeping GIDs, timestamps, and actions intact over narrative text.
    You are rewriting a video summary using the original summary and detection metadata.
    """
    return types.Content(role="user", parts=[types.Part(text=prompt)])

def make_user_message_recursive(summaries):
    prompt = f"""
    Task:
    You are merging these intermediate summaries into one consolidated summary.
    Batches of summaries:
    {' '.join(summaries)}
    
    Do NOT compress or abstract further. Your goal is to combine all details exactly as given.
    
    Rules:
    1. Preserve ALL GIDs, timestamps, actions, and context from input summaries.
    2. Merge duplicate GIDs by combining their timeframes and actions chronologically.
    3. Keep all sections intact:
    **Overall Summary**
    **Detailed GID Actions**
    **Key Transitions and Observations**
    **Suspicious Activity Note**
    **Master GID List**
    4. If input summaries are already condensed, DO NOT remove any detail—just merge.
    5. Output must follow the same structure as above.
    """
    return types.Content(role="user", parts=[types.Part(text=prompt)])

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

    filter_expr = f'metadata["mode"]=="text" AND metadata["video_path"]=="Copy_of_D02_20250918150609_001_16mins.mp4" AND metadata["updated_summary"] IS NOT NULL'
   
    print("Querying Milvus collection...")
    collection_data = milvus_manager.query(
        collection_name=args.collection_name,
        filter=filter_expr,
        limit=1000,
        output_fields=["pk", "metadata", "vector"]
    )

    print(f"Total items retrieved from DB: {len(collection_data)}")

    if not collection_data:
        print("No data found in the collection. Skipping agent invocation.")
        return [] 
    
    session_service = InMemorySessionService()
    user_id = "user1"
    session_id = "session1"
    app_name = "video_summary_app"

    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=summarizer_agent, app_name=app_name, session_service=session_service)

    # Initial batch-wise summarization of DB summaries
    intermediate_summaries = []
    batch_count = 0
    for batch in query_summaries_from_db(collection_data, batch_size=5):
        batch_count += 1
        print(f"Processing batch {batch_count} with {len(batch)} summaries")
        print(f"  batch summaries {batch} ")
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
        for i in range(0, len(intermediate_summaries), 5):
            batch = intermediate_summaries[i:i+5]
            print(f"  Summarizing batch {i//5 + 1} of size {len(batch)}")
            print(f"  batch summaries {batch} ")
            input_msg = make_user_message_recursive(batch)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="tracking_logs")
    
    args = parser.parse_args()
    asyncio.run(run_recursive_summarization(args))
