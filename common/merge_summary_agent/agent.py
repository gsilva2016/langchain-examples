import os
import asyncio
import litellm
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
# from .tools import *
from dotenv import load_dotenv

# Load ENV settings
load_dotenv()
agent_model_endpoint = os.getenv("AGENT_MODEL_ENDPOINT", "http://localhost:8012/v3")
agent_model_name = os.getenv("AGENT_MODEL_NAME", "")
agent_instruction = os.getenv("AGENT_INSTRUCTION", "")

# Enable DEBUG
#litellm._turn_on_debug()

# Connecct model server endpoint
llm_serving = LiteLlm(
    model = agent_model_name,
    api_base = agent_model_endpoint,
    api_key="none",
    additional_drop_params=["extra_body"]
)

# Summarizer agent that summarizes a list of text chunks
summarizer_agent = Agent(
    name="summarizer_agent",
    model=llm_serving,
    description="Summarizes a list of video chunk summaries.",
    instruction=("""
        Merge multiple video summaries into one unified summary.
        Preserve all structured details (GIDs, timestamps, actions) and follow the detailed merging rules.
        Return only the consolidated summary in the required format.
        """
	)
)
