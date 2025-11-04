import logging
from typing import AsyncGenerator
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from .agent import classifier_agent,metadata_agent,rag_agent
from google.genai import types
import re

logger = logging.getLogger(__name__)

def extract_final_label_from_cot(output_text):
    # Split reasoning/think blocks
    parts = re.split(r"</?think>", output_text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        # Take text after last </think>
        final_text = parts[-1].strip()
        # Search for classification keyword only in final part
        match = re.search(r"(tracking_logs|individual_report|rag)", final_text)
        if match:
            return match.group(1)
    # Fall back to a safer default if no match found
    return "rag"

class QueryRouterAgent(BaseAgent):
    classifier_agent: LlmAgent
    metadata_agent: LlmAgent
    rag_agent: LlmAgent

    def __init__(self, 
                 name: str,
                 classifier_agent: LlmAgent, 
                 metadata_agent: LlmAgent,
                 rag_agent: LlmAgent):
        sub_agents = [classifier_agent, metadata_agent, rag_agent]
        super().__init__(name=name, 
                         classifier_agent=classifier_agent,
                         metadata_agent=metadata_agent,
                         rag_agent=rag_agent,
                         sub_agents=sub_agents)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting query routing workflow.")
        output_text = ""
        async for event in self.classifier_agent.run_async(ctx):
            if event.content and event.content.parts:
                output_text += event.content.parts[0].text
        print("output_text:", output_text)
        # match = re.search(r"(tracking_logs|individual_report|rag)", output_text)
        # collection_name = match.group(1) if match else None

        collection_name = extract_final_label_from_cot(output_text)
        if collection_name not in {"tracking_logs", "individual_report", "rag"}:
            collection_name = "rag" ## fallback
        ctx.session.state["collection_name"] = collection_name
        
        # Yield a synthetic final event with clean classification only
        yield Event(
            author=self.name,
            content=types.Content(
                role="system",
                parts=[types.Part(text=collection_name)]
            )
        )
        # Then continue routing to other agents...
    
        # Debug log
        logger.info(f"[{self.name}] Classifier returned collection_name='{collection_name}'")

        # Conditionally call metadata or rag agent based on classification
        if collection_name in ["tracking_logs", "individual_report"]:
            logger.info(f"[{self.name}] Routing to MetadataAgent...")
            async for event in self.metadata_agent.run_async(ctx):
                yield event
        elif collection_name == "rag":
            logger.info(f"[{self.name}] Routing to RagAgent...")
            async for event in self.rag_agent.run_async(ctx):
                yield event
        else:
            logger.warning(f"[{self.name}] Unknown collection_name '{collection_name}', no further action.")

        logger.info(f"[{self.name}] Query routing workflow completed.")
