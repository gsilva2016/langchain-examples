import os
import asyncio
import litellm
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
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
regenerate_summary_agent = Agent(
    name="enhance_summary_agent",
    model=llm_serving,
    description="Summarizes a list of video chunk summaries.",
    instruction = """
        Rewrite video summaries based on provided original summary and detection metadata.
        Follow the user-provided prompt for detailed rules.
        Return only the rewritten summary.
      """
      )