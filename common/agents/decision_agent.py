from google.adk.agents import LlmAgent
from pydantic import BaseModel
from google.adk.models.lite_llm import LiteLlm
import os
from dotenv import load_dotenv
load_dotenv()
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama')

# Define Pydantic input schema
class RecaptionDecisionInput(BaseModel):
    summary: str
    score: float

model_string = "openai/" + LLAMA_MODEL
    
ovms_serving = LiteLlm(
    model=model_string,
    api_base="http://localhost:8013/v3/", 
    api_key="none"
)

# vLLM_serving=LiteLlm(
     # model="hosted_vllm/llmware/llama-3.2-3b-instruct-ov", 
     # api_base="http://localhost:8001/v1", 
     # api_key="not-needed"
# )
decision_agent = LlmAgent(
    model=ovms_serving,
    name="video_caption_analyzer",
    description="Analyzes video caption summary and score to determine if review is required.",
    instruction="""
You are a security assistant. Given a video summary and a score (0.0 = safe, 1.0 = suspicious), decide if the situation is alarming.
If the score is 0.8 or higher, or if the summary describes suspicious behavior, set "review_required" to true and provide a brief reason.
Otherwise, set "review_required" to false. Respond in JSON like:
{"review_required": true, "reason": "Suspicious loitering detected."}
or
{"review_required": false}
""",
    input_schema=RecaptionDecisionInput,
    output_key="review_decision"
)