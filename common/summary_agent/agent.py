import os
import asyncio
import litellm
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from .tools import *
from dotenv import load_dotenv
from google.adk.tools import FunctionTool


# Load ENV settings
load_dotenv()
agent_model_endpoint = os.getenv("AGENT_MODEL_ENDPOINT", "http://localhost:8012/v3")
agent_model_name = os.getenv("AGENT_MODEL_NAME", "")
agent_instruction = os.getenv("AGENT_INSTRUCTION", "")

# Enable DEBUG
# litellm._turn_on_debug()

# Connecct model server endpoint
llm_serving = LiteLlm(
    model = agent_model_name,
    api_base = agent_model_endpoint,
    api_key="none",
    additional_drop_params=["extra_body"]
)
# Register this as a tool for your agent
report_tool = FunctionTool(generate_end_of_day_report)

# Create an agent which can use multiple tools to take its action. 
# Note ADKWEB requires root_agent object
root_agent = Agent(
    name = "report_agent",
    model = llm_serving,
    description = "AI agent to generate end of day report",
    instruction = agent_instruction,
    tools=[report_tool]
)
