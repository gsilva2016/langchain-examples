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
    # """
# You are a retrieval agent.
# Your primary task is to handle semantic search queries when the routing decision from the classifier specifies 'rag'.

# Steps:
# 1. Check if the input variable {collection_name} equals 'rag'.
# 2. If yes, call the rag_retriever_tool with the current session state (accessible via tool_context.state),
   # passing in relevant fields like query, filter_expression, query_img, and Milvus configuration.
# 3. Return the retriever output directly (no extra commentary).

# If {collection_name} is not 'rag', do nothing.
# """,
    tools=[rag_retriever_tool]
)
# router_pipeline = SequentialAgent(
    # name="RouterPipeline",
    # sub_agents=[classifier_agent, metadata_agent, rag_agent]
# )