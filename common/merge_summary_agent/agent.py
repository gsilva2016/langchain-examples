import os
import asyncio
import litellm
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
# from .tools import *
from dotenv import load_dotenv
from google.adk.tools import agent_tool

#from google.adk.tools.agent_tool import AgentTool
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
    instruction=(
        "You are responsible for summarizing batches of video summaries."
        "Each batch contains up to 4 summaries."
        "Your task is to generate a single, concise but comprehensive overall summary that synthesizes the information from all summaries in the batch."
        "The scene takes place in a warehouse."
        "Rules:"
        "- Include ALL individuals: Every Global ID (GID) mentioned in the input must appear in the final summary."
        "- For each GID, include its consolidated time interval and observed actions (entering, exiting, stationary, interacting with objects or people)."
        "- Merge duplicate GIDs across summaries into one combined description."
        "- Highlight key behaviors, transitions between store areas, and interactions with objects or people."
        "- Avoid repeating individual summaries verbatim; instead, create one unified narrative."
        "- Ensure completeness: Before finalizing, verify that all GIDs from the input are present in the output."
        "- Maintain clarity and brevity without losing important details."
    	# "You are responsible for summarizing batches of video summaries. "
    	# "Each batch contains up to 4 summaries. "
    	# "Your task is to generate a single, concise but comprehensive overall summary "
    	# "that synthesizes the information from all summaries in the batch. "
        # "The scene takes place in a warehouse."
        # "Focus on the activities and movements of **all visible individuals** or observed persons in the store. "
        # "Include All Individuals: Mention every detected Global ID and their consolidated presence interval."
    	# # "Make sure in every summary you include all persons and their actions. "
    	# "Highlight key behaviors, transitions between store areas, interactions with objects or people, "
    	# "and any movements that may indicate intent or purpose. "
    	# "Avoid listing or repeating the individual summaries. Instead, combine the key details into one unified narrative. "
    	# "Ensure the summary reflects the full scope of observed activity, maintaining clarity and brevity without losing important details."
	)
)

# # Wrap the summarizer as a tool
# summarizer_tool = agent_tool.AgentTool(agent=summarizer_agent)

# # # Manager agent orchestrates recursively using the summarizer tool
# # manager_agent = Agent(
    # # name="manager_agent",
    # # model=llm_serving,
    # # description="Dynamically recursively calls summarizer tool on batches until final summary.",
    # # instruction=(
        # # "You are responsible for summarizing a list of video summaries. "
        # # "If there is more than one summary, group them into batches of 4, "
        # # "call the summarizer tool on each batch to get intermediate summaries, "
        # # "then repeat recursively on the list of intermediate summaries. "
        # # "Once a single summary remains, return it as the final summary."
    # # ),
    # # tools=[summarizer_tool]
# # )

# # Manager agent that simply invokes the summarizer tool per batch
# manager_agent = Agent(
    # name="manager_agent",
    # model=llm_serving,
    # description="Handles summarization requests by calling the summarizer tool on each batch.",
    # instruction=(
        # "You are responsible for summarizing batches of video summaries. "
        # "Each batch contains up to 4 summaries. Use the summarizer tool to summarize each batch. "
        # "You do not need to perform recursive summarization; that will be handled externally."
    # ),
    # tools=[summarizer_tool]
# )




    # "You receive a list of summaries. If the list length is greater than 1, split into batches of 4, "
        # "call the summarizer tool on each batch, then repeat this recursively on the new intermediate summaries. "
        # "When only one summary remains, return it as the final summary."
