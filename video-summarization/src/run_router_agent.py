from common.router_agent.agent import classifier_agent,metadata_agent,rag_agent ##router_pipeline
from common.router_agent.customagent import QueryRouterAgent, classifier_agent, metadata_agent, rag_agent
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from common.milvus.milvus_wrapper import MilvusManager
import argparse
import asyncio
import os

router_agent = QueryRouterAgent(
    name="QueryRouterAgent",
    classifier_agent=classifier_agent,
    metadata_agent=metadata_agent,
    rag_agent=rag_agent
)

async def adk_runner(args):
    milvus_manager = MilvusManager()    
    session_service = InMemorySessionService()
    user_id = "user"
    session_id = "session"
    app_name = "query_router"
    initial_state = {
        "milvus_manager": milvus_manager,
        "query": args.query_text,
        "milvus_uri": args.milvus_uri,
        "milvus_port": args.milvus_port,
        "milvus_dbname" : args.milvus_dbname
    }
 
    # Create session asynchronously, with  initial state
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )
    
    runner = Runner(agent=router_agent, ##router_pipeline, 
        app_name=app_name, 
        session_service=session_service)
            
    
    user_input = types.Content(
        role='user',
        parts=[types.Part(text=args.query_text)] ##"what happned in the viode?")] ##Show me idle time last week")]
    )
   
    
    response_events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
    
    # async for event in response_events:
        # print("Agent output:", event.content.parts[0].text if event.content else None)
        # if event.is_final_response():
            # final_answer = event.content.parts[0].text if event.content else "No final response content"
            # print("Final agent response:", final_answer)
            # break
    async for event in response_events:
        if event.content and event.content.parts:
            print("Agent output:", event.content.parts[0].text)
        if event.is_final_response():
            final_answer = event.content.parts[0].text
            last_final_answer = final_answer  # keep updating
    # after loop
    print("Final agent response:", last_final_answer)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, default="rag")
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="video_chunks")
    args = parser.parse_args()
    
    # load_dotenv()
    interval_minutes = float(os.getenv("RUNNER_INTERVAL_MINUTES", 15))  # Default is 15 min
    interval_seconds = int(interval_minutes * 60)

    asyncio.run(adk_runner(args))


