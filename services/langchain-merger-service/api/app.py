import argparse
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from langchain_summarymerge_score import SummaryMergeScoreTool
from pydantic import BaseModel


class SummaryMergerRequest(BaseModel):
    """
    Pydantic model for the request body of the merge_summaries endpoint. It expects a dictionary of summaries.
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }
    """
    summaries: dict


class SummaryMergerResponse(BaseModel):
    """
    Pydantic model for the response body of the merge_summaries endpoint.
    It will return the overall summary and the anomaly score.
    """
    overall_summary: str
    anomaly_score: float


app = FastAPI()
summary_merger = None


@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello from App. Use /merge_summaries/<summaries dict {'summaries': {'chunk0': 'text'...}}> to perform summary merging/assign anomaly score"}


@app.post("/merge_summaries")
def merge_summaries(request: SummaryMergerRequest):
    """
    Endpoint for calling summary merger. Input should be in this format:
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }"""
    
    print(f"Received request to merge summaries and running {summary_merger.name}")
  
    # output = summary_merger.merge_summaries(request.summaries)
    # print(f"Summaries to merge: {request.summaries}")
    output = summary_merger.invoke({"summaries": request.summaries})
    return SummaryMergerResponse(**output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--model_path", required=True, help="Path to LLAMA model")
    parser.add_argument("--device", nargs="?", default="GPU")
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--endpoint_uri", nargs="?", default="0.0.0.0")
    parser.add_argument("--endpoint_port", default=8000, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        sys.exit(1)
        
    if not args.endpoint_uri == "" and not args.endpoint_port:
        print("FastAPI server running on default localhost:8000.")
        
    summary_merger = SummaryMergeScoreTool(
        model_id=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens        
    )

    uvicorn.run(app, host=args.endpoint_uri, port=args.endpoint_port)
