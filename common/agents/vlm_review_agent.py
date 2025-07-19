from google.adk.agents import LlmAgent
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')

class ReviewDecisionInput(BaseModel):
    review_required: bool
    reason: str = None  # Optional, only present if review_required is true

vlm_review_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="vlm_video_caption_agent",
    description="If review_required is true, generate a new caption for the provided video frame and an anomaly score.",
    instruction="""
You are an expert investigator. Review the attached video frame from a shopping aisle security camera.

If 'review_required' is true:
- Carefully observe the video for any moments of identified or highly suspected shoplifting or stealing.
- Focus on people interacting with objects on display and taking items into their possession.
- Provide a score between 0 and 1 to represent the amount of suspicious activity observed (0 = no suspicious activity, 1 = highly suspicious activity). If no humans are visible, the score must be 0.
- The score should be a float rounded to the tenth decimal.

**Respond ONLY in the following JSON format. Do not include any extra text, explanations, or markdown. Only output the JSON object.**

{
  "overall_summary": "A summary of the entire video in about five sentences or less, focused on the people rather than the scene itself.",
  "potential_suspicious_activity": "List any activities that might indicate suspicious behavior.",
  "anomaly_score": 0.0
}

If 'review_required' is false, respond with: {"overall_summary": "", "potential_suspicious_activity": "", "anomaly_score": 0.0}
""",
    # project=project,
    # location=location,
    input_schema=ReviewDecisionInput,
    output_key="vlm_caption"
)