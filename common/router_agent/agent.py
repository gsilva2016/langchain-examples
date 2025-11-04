import os
import asyncio
import litellm
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from .tools import *
from dotenv import load_dotenv
from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent, SequentialAgent


# Load ENV settings
load_dotenv()
agent_model_endpoint = os.getenv("AGENT_MODEL_ENDPOINT", "http://localhost:8012/v3")
agent_model_name = os.getenv("AGENT_MODEL_NAME", "")
query_classifier_agent_instruction = os.getenv("QUERY_CLASSIFIER_AGENT_INSTRUCTION", "")
metadata_agent_instruction = os.getenv("METADATA_AGENT_INSTRUCTION", "")
rag_agent_instruction = os.getenv("RAG_AGENT_INSTRUCTION", "")

# Enable DEBUG
# litellm._turn_on_debug()

# Connecct model server endpoint
llm_serving = LiteLlm(
    model = agent_model_name,
    api_base = agent_model_endpoint,
    api_key="none",
    additional_drop_params=["extra_body"]
)
    
metadata_extractor_tool = FunctionTool(metadata_extractor)
rag_retriever_tool = FunctionTool(rag_retriever)

classifier_agent = LlmAgent(
    name="QueryClassifier",
    model=llm_serving,
    instruction=query_classifier_agent_instruction,
    output_key="collection_name"
)

metadata_agent = LlmAgent(
    name="MetadataAgent",
    model=llm_serving,
    instruction=metadata_agent_instruction,
    tools=[metadata_extractor_tool],
    output_key="metadata_output"
)


rag_agent = LlmAgent(
    name="RagAgent",
    model=llm_serving,
    instruction=rag_agent_instruction,
    tools=[rag_retriever_tool]
)
