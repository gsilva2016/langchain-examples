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

NUMBER_OF_INITIAL_SUMMARIES = 4
NUMBER_OF_RECURSIVE_SUMMARIES = 3

def query_summaries_from_db(collection_data, batch_size=NUMBER_OF_INITIAL_SUMMARIES):
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
    You are responsible for summarizing these batches of video summaries. Each batch contains up to {NUMBER_OF_INITIAL_SUMMARIES} summaries.
    
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
    
def make_user_message_recursive(summaries, include_pre_merge_enumeration: bool = False):
    """
    Build a robust user message instructing the model to merge multiple intermediate summaries
    into ONE consolidated summary without losing any details.

    Parameters
    include_pre_merge_enumeration : bool    ----------
        If True, the prompt asks the model to first enumerate all GIDs and timestamps across inputs
        before producing the merged output (strong guardrail against omissions).

    Returns
    -------
    types.Content
        A content object containing the prompt text.
    """
    # Wrap and separate each summary with a strong delimiter
    wrapped_blocks = [
        f"### Summary {i}\n{s.strip()}"
        for i, s in enumerate(summaries, start=1)
    ]
    delimiter = "\n\n--- SUMMARY DELIMITER ---\n\n"
    joined = delimiter.join(wrapped_blocks)

    # Optional pre-merge enumeration section for stronger coverage guarantees
    pre_merge_section = ""
    if include_pre_merge_enumeration:
        pre_merge_section = """
        Pre-Merge Enumeration (brief, before merging):
        - List all GIDs appearing across input summaries.
        - List all distinct timestamps/ranges appearing across input summaries.
        Then proceed with the merged output.
        """

    prompt = f"""
    Task:
    You will merge the following intermediate summaries into ONE comprehensive, consolidated summary.
    
    Input Summaries (each separated by a delimiter):
    {joined}
    
    Important:
    - Treat each delimited block as a distinct source that MUST be fully integrated.
    - Do NOT compress, paraphrase, or omit any detail. Copy and merge ALL information from ALL blocks.
    
    Rules:
    1) Preserve ALL identifiers and metadata:
    - GIDs
    - Timestamps (ranges and single points)
    - Actions and interactions
    - Context and observations
    
    2) Merge duplicate GIDs:
    - Combine their timeframes and actions in strict chronological order (earliest to latest).
    - Keep overlapping/conflicting entries; note discrepancies rather than dropping them.
    
    3) Maintain this exact output structure (no extra sections, no missing sections):
    **Overall Summary**
    **Detailed GID Actions**
    **Key Transitions and Observations**
    **Suspicious Activity Note**
    **Master GID List**
    
    4) Chronological completeness:
    - Include events from the very beginning of the first summary through the very end of the last summary.
    - Do NOT skip any later chunks or leave gaps.
    - If any timestamps are open-ended (e.g., "06:02–"), carry them forward when possible and keep them as-is if unresolved.
    
    5) Unify sections across summaries:
    - Combine ALL "Overall Summary" texts into one unified description (no omissions).
    - Merge ALL "Detailed GID Actions" into a single list sorted by timestamp.
    - Merge ALL "Key Transitions and Observations" into one section.
    - Merge ALL "Suspicious Activity Note" content into one section.
    - Merge ALL "Master GID List" entries into one deduplicated list.
    
    6) Verification before finalizing:
    - Ensure every GID appearing in the input summaries also appears in **Master GID List**.
    - Ensure every timestamp (including ranges and single points) appears somewhere in **Detailed GID Actions**.
    - Ensure events from the last input block (latest timestamps) appear near the end of **Detailed GID Actions**.
    7) For overlapping or repeated time ranges (e.g., 03:40–08:20 and 03:40–10:40), include ALL entries; do not collapse or rewrite them.
    8) Every GID in the Master GID List MUST have at least one entry in **Detailed GID Actions** (even if minimal).
    
    {pre_merge_section}
    
    Output:
    Return ONE merged summary that strictly follows the structure above, is exhaustive, and preserves all details unchanged.
    """.strip()

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

    filter_expr = f'metadata["mode"]=="text" AND metadata["updated_summary"] IS NOT NULL'
   
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
    for batch in query_summaries_from_db(collection_data, batch_size=NUMBER_OF_INITIAL_SUMMARIES):
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
        for i in range(0, len(intermediate_summaries), NUMBER_OF_RECURSIVE_SUMMARIES):
            batch = intermediate_summaries[i:i+NUMBER_OF_RECURSIVE_SUMMARIES]
            print(f"  Summarizing batch {i//NUMBER_OF_RECURSIVE_SUMMARIES + 1} of size {len(batch)}")
            print(f"  batch summaries {batch} ")
            input_msg = make_user_message_recursive(batch, include_pre_merge_enumeration=True)
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
