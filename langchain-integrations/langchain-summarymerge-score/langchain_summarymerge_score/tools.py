"""SummaryMergeScore tools."""

import ast
from concurrent.futures import ThreadPoolExecutor
import math
import os
import re
import sys
import time
from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import requests
import openvino_genai as ov_genai
from dotenv import load_dotenv

class SummaryMergeScoreToolInput(BaseModel):
    """Input schema for SummaryMergeScore tool.

    This docstring is **not** part of what is sent to the model when performing tool
    calling. The Field default values and descriptions **are** part of what is sent to
    the model when performing tool calling.
    """
    summaries: dict = Field(..., description="Dictionary of summaries to merge")
        
class SummaryMergeScoreTool(BaseTool):  # type: ignore[override]
    """SummaryMergeScore tool.

    Setup:
        Install ``langchain-summarymerge-score``.

        .. code-block:: bash

            pip install -U langchain-summarymerge-score
            Set HUGGINGFACE_TOKEN via `export HUGGINGFACE_TOKEN=<YOUR_ACCESS_TOKEN>`

    Instantiation:
    --- Via OVMS endpoint server (local or remote) ---
    First ensure you have the OpenVINO Model Server (OVMS) running with the LLM model loaded.
    Then instantiate the tool with the model ID and OVMS endpoint.
    
        .. code-block:: python
            from langchain_summarymerge_score import SummaryMergeScoreTool

            tool = SummaryMergeScoreTool(
                model_id="meta-llama/Llama-3.2-3B-Instruct",
                api_base="http://localhost:8013/v3/chat/completions", # your OVMS endpoint
                device="GPU", # CPU, GPU, or NPU
                max_new_tokens=512,
                batch_size=5,
            )

    Invocation with args:
        .. code-block:: python

            summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

            output = tool.invoke({"summaries": summaries})

        .. code-block:: python

            {"overall_summary": "Merged summary text", "anomaly_score": 0.5}
    
    --- Via local model directory ---    
        .. code-block:: python
            from langchain_summarymerge_score import SummaryMergeScoreTool

            tool = SummaryMergeScoreTool(
                model_id="my_meta_llama_3.2_3B_Instruct/", # path to local model directory
                device="GPU", # CPU, GPU, or NPU
                max_new_tokens=512,
                batch_size=5,
            )

    Invocation with args:
        .. code-block:: python

            summaries = {
            "summaries": {
                "chunk_0": "text1",
                "chunk_1": "text2"
                }
            }

            output = tool.invoke({"summaries": summaries})

        .. code-block:: python

            {"overall_summary": "Merged summary text", "anomaly_score": 0.5}            
            
    """  # noqa: E501

    name: str = "Summary Merge Score Tool"
    """The name that is passed to the model when performing tool calling."""
    description: str = "This tool merges summaries using a specified model and device."
    """The description that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = SummaryMergeScoreToolInput
    """The schema that is passed to the model when performing tool calling."""
    
    api_base: str = None
    device: str = "GPU"
    max_new_tokens: int = 512
    batch_size: int = 8
    ov_llm: object = None
    summary_prompt: str = None
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    ov_pipe: object = None

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "GPU", 
                 max_new_tokens: int = 512, 
                 batch_size: int = 8,
                 env_file: str = ".env",
                 api_base: str = None,
                 ov_pipe: object = None):
        super().__init__()
        
        load_dotenv(env_file)
        
        hf_token_access_token = os.getenv("HUGGINGFACE_TOKEN", None)
        if hf_token_access_token is None:
            print("HUGGINGFACE_TOKEN not found in .env file. Please set it to access gated models.")
            print("For more information on user access tokens for access to gated models see https://huggingface.co/docs/hub/en/security-tokens")
            sys.exit(1)
            
        self.model_id = os.getenv("LLAMA_MODEL", model_id)
        self.device = os.getenv("SUMMARY_MERGER_LLM_DEVICE", device)

        self.api_base = api_base
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        default_prompt = """Write a response that appropriately completes the request.
        ### Instruction: Your job is to merge multiple chunk summaries into one balanced and complete summary. How would you rate the scene described on a scale from 0.0 to 1.0, with 0.0 representing a standard scene and 1.0 denoting a scene with suspicious activities?
        Please organize your answer according to this example:
        Overall Summary: <summary here>
        Activity Observed: <bullet points>
        Potential Suspicious Activity: <suspicious behavior if any>
        Number of people in the scene: <best estimate. DO NOT overcount>.
        Anomaly Score: <float from 0 to 1, based on severity of suspicious activity>.
        """

        self.summary_prompt = os.getenv("SUMMARY_PROMPT", default_prompt)

        # if device is CPU or GPU
        if self.device in ["CPU", "GPU"]:
            self.batch_size = batch_size
        else:
            # 2 is the max it could handle
            self.batch_size = 2
        
        if self.api_base is None:
            if not os.path.exists(self.model_id):
                print(f"Model path {self.model_id} does not exist. Please provide a valid model path.")
                sys.exit(1)
            
            # if device is CPU or GPU
            if self.device in ["CPU", "GPU"]:
                pipeline_config = {"CACHE_DIR": f"./cache/llm_{self.device.lower()}", "PERFORMANCE_HINT": "LATENCY"}
            else:
                # for NPU, we need to set GENERATE_HINT to BEST_PERF, this option is not available for GPU
                pipeline_config = {"CACHE_DIR": f"./cache/llm_{self.device.lower()}", "MAX_PROMPT_LEN": 1500, "MIN_RESPONSE_LEN": self.max_new_tokens, "GENERATE_HINT": "BEST_PERF"}

            self.ov_pipe = ov_genai.LLMPipeline(self.model_id, device=self.device, **pipeline_config)

        print(f"Running model: {self.model_id} on device: {self.device}  batch size: {self.batch_size} max_new_tokens: {self.max_new_tokens}")    
        
    def _run(
        self, summaries: dict, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Merge summaries generated from multiple chunks of text and generate a final summary with an anomaly score
        """
        start_time = time.time()
        chunks = list(summaries.values())

        num_batches = math.ceil(len(chunks) / self.batch_size)
        print(f"Num of batches to process: {num_batches}")

        batch_summaries = []
        
        for i in range(num_batches):
            print("--------------------------------------------")
            batch_texts = chunks[i * self.batch_size:(i + 1) * self.batch_size]
            print(f"Processing batch {i + 1}... having {len(batch_texts)} chunks")
            batch_summary = self.summarize_batch(batch_texts)
            batch_summaries.append(batch_summary)

        # recursively merge summaries which are greater than batch size
        while len(batch_summaries) > self.batch_size:
            print(f"Recursively merging summaries, current batch size: {len(batch_summaries)}")
            temp = []
            for i in range(0, len(batch_summaries), self.batch_size):
                group = batch_summaries[i: i + self.batch_size]
                print(f"Processing batch... having {len(group)} chunks")
                temp.append(self.summarize_batch(group))
            batch_summaries = temp

        print("--------------------------------------------")
        print(f"Processing final batch of having {len(batch_summaries)} chunks")

        # if multiple summaries are present, merge them, else use the single summary
        if len(batch_summaries) > 1:
            final_summary = self.summarize_batch(batch_summaries)
        else:
            print("Final batch has only one chunk present, no need to merge further.")
            final_summary = batch_summaries[0]

        # extract anomaly score from final summary using a regex pattern
        final_anomaly_score = self.extract_anomaly_score(final_summary)
        print(
            f"Time taken for merge-summarize {len(summaries)} chunk summaries: {time.time() - start_time:.2f} seconds")

        return {"overall_summary": final_summary, "anomaly_score": final_anomaly_score}

    def summarize_batch(self, texts):
        """
        Summarize a batch of summaries using the chosen model
        """
        text = "\n\n".join(texts)

        if self.api_base is not None:
            data = {
                "model": self.model_id,
                "max_tokens": self.max_new_tokens,
                "temperature": 0,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": self.summary_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Now do the same for:\n{text}",
                    },
                ]
            }

            response = requests.post(self.api_base, 
                                    json=data, 
                                    headers={"Content-Type": "application/json"})

            if response.status_code == 200:
                output_json = response.json()
                output_text = output_json["choices"][0]["message"]["content"]
                print("Response JSON:", output_json)
            else:
                print("Error:", response.status_code, response.text)
                return
        else:
            if self.device in ["CPU", "GPU"]:
                config = ov_genai.GenerationConfig()
                config.max_new_tokens = self.max_new_tokens
                output_text = self.ov_pipe.generate(self.summary_prompt.format(question=text), config=config)
            else:
                output_text = self.ov_pipe.generate(self.summary_prompt.format(question=text))

        return output_text.strip()

    @staticmethod
    def extract_anomaly_score(summary):
        # matching based on multiple scenarios observed; goal is to match floating point or integer after Anomaly Score
        # Anomaly Score sometimes is encapsulated within ** and sometimes LLM omits
        match = re.search(r"Anomaly Score:?\s*(-?\d+(\.\d+)?)", summary, re.DOTALL)
        if match:
            return float(match.group(1)) if match.group(1) else 0.0
        return 0.0