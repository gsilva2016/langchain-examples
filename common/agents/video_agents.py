import json
import uuid
import asyncio
import re
import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from common.agents.decision_agent import decision_agent
from common.agents.vlm_review_agent import vlm_review_agent


from dotenv import load_dotenv
load_dotenv()

async def run_video_recaption_agent(summary, score, video_path):
    # Create session service and session
    session_service = InMemorySessionService()
    APP_NAME = "video_caption_app"
    USER_ID = "user_001"
    SESSION_ID = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={},
    )
    
    
    item = {
        "overall_summary": summary,
        "anomaly_score": score
    }
    
    print("run_video_recaption_agent input overall_summary and anomaly_score : \n", summary, score)
    text_message = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps(item))]
    )

    # 1. Run text-only agent (LiteLLM)
    runner1 = Runner(
        agent=decision_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    review_decision = None
    for event in runner1.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=text_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                review_decision = event.content.parts[0].text
    # Clean up markdown code fences and whitespace
    def clean_llm_json(s):
        # Remove all lines that are only triple backticks or triple backticks with 'json'
        return re.sub(r'(?m)^```(?:json)?\s*$', '', s).strip()

    
    print("Raw review_decision:", review_decision)
    review_decision = clean_llm_json(review_decision)
    # print("Cleaned review_decision:", review_decision)
    try:
        review_decision_json = json.loads(review_decision)
        print("Parsed review_decision_json:", review_decision_json)
    except Exception as e:
        print("JSON parsing failed:", e)
        review_decision_json = {}
    

    # Only run VLM agent if review is required
    if review_decision_json.get("review_required"):
        # print("review is needed")
        # Prepare video for Gemini
        with open(video_path, "rb") as vid_file:
            video_bytes = vid_file.read()
        vlm_message = types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")),
                types.Part(text=json.dumps(item))
            ]
        )
        
        
        runner2 = Runner(
            agent=vlm_review_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        vlm_result = None
        for event in runner2.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=vlm_message,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    vlm_result = event.content.parts[0].text
        
        vlm_result = clean_llm_json(vlm_result)
        return review_decision, vlm_result
    else:
        return review_decision, None



    # (Optional) Print session state
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\n=== Final Session State ===")
    for key, value in session.state.items():
        print(f"{key}: {value}")
