import os
from queue import Queue
import time
from dotenv import load_dotenv
from langchain_summarymerge_score import SummaryMergeScoreTool
import numpy as np
import requests
from PIL import Image

load_dotenv()
SUMMARY_MERGER_ENDPOINT = os.environ.get("SUMMARY_MERGER_ENDPOINT", None)

def send_summary_request(summary_q: Queue):
    while True:
        chunk_summaries = []
        
        while not summary_q.empty():
            chunk_summaries.append(summary_q.get())
            
        if chunk_summaries:
            formatted_req = {
                "summaries": {chunk["chunk_id"]: chunk["chunk_summary"] for chunk in chunk_summaries}

            }
            print(f"Summary Merger: Sending {len(chunk_summaries)} chunk summaries for merging")
            try:
                summary_merger = SummaryMergeScoreTool(
                    api_base=SUMMARY_MERGER_ENDPOINT,
                )
                merge_res = summary_merger.invoke(formatted_req)
                
                print(f"Overall Summary: {merge_res['overall_summary']}")
                print(f"Anomaly Score: {merge_res['anomaly_score']}")

            except Exception as e:
                print(f"Summary Merger: Request failed: {e}")
            
            return merge_res
        else:
            print("Summary Merger: Waiting for chunk summaries to merge")
    
        time.sleep(30)
    
def ingest_frames_into_milvus(frame_q: Queue, milvus_manager: object):
    while True:
        frames = []
        
        while not frame_q.empty():
            frames.append(frame_q.get())
        
            print(f"Milvus: Ingesting {len(frames)} frames into Milvus")
            try:
                if frames:
                    response = milvus_manager.embed_img_and_store(frames)
                
                print(f"Milvus: Chunk Frames Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
            
            except Exception as e:
                print(f"Milvus: Frame Ingestion Request failed: {e}")
        else:
            print("Milvus: Waiting for chunk frames to ingest")
    
        time.sleep(10)

def ingest_summaries_into_milvus(ingest_q: Queue, milvus_manager: object):
    while True:
        chunk_summaries = []
        
        while not ingest_q.empty():
            chunk_summaries.append(ingest_q.get())
        
            print(f"Milvus: Ingesting {len(chunk_summaries)} chunk summaries into Milvus")
            try:
                if chunk_summaries:
                    response = milvus_manager.embed_txt_and_store(chunk_summaries)
                
                print(f"Milvus: Chunk Summaries Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
            
            except Exception as e:
                print(f"Milvus: Chunk Summaries Ingestion Request failed: {e}")
        else:
            print("Milvus: Waiting for chunk summaries to ingest")
    
        time.sleep(30)

def search_in_milvus(query_text: str, milvus_manager: object):
    try:
        
        response = milvus_manager.search(query=query_text)
        print(response.content)
        
        return response
    
    except Exception as e:
        print(f"Search in Milvus: Request failed: {e}")

def query_vectors(expr: str, milvus_manager: object, collection_name: str = "chunk_summaries"):
    try:
        response = milvus_manager.query(expr=expr, collection_name=collection_name)
        
        return response
    
    except Exception as e:
        print(f"Query Vectors: Request failed: {e}")
        

def get_sampled_frames(frame_q: Queue):        
    chunk_id = 1
    
    # change below to rotate between two images
    while True:
        # generate random array
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
        sampled = []
        for i in range(0, 30):
            sampled.append(frame)

        random_chunk = {
            "video_id": chunk_id,
            "chunk_id": f"chunk_{chunk_id}_{chunk_id}",
            "chunk_path": f"video_chunk_path",
            "frames": sampled,
            "start_time": f"random time",
            "end_time": f"random time"
        }
        
        print(f"Generating chunk for chunk {chunk_id}")
        
        frame_q.put(random_chunk)
        
        chunk_id += 1
        time.sleep(10)

    
def generate_chunk_summaries(i_queue, s_queue, f_queue):
    # To be replaced to logic to call miniCPM service
    chunk_id = 1
    while True:
        time.sleep(15)
        
        random_chunk = {
            "video_id": chunk_id,
            "chunk_id": f"chunk_camera_{chunk_id}_{chunk_id}",
            "chunk_path": f"video_chunks/cam_{chunk_id}/chunk_{chunk_id}.mp4",
            "chunk_summary": f"Writing some random data here for chunk {chunk_id}.",
            "start_time": f"random time",
            "end_time": f"random time"
        }
        
        print(f"Generating chunk summary for chunk {chunk_id}")
        
        i_queue.put(random_chunk)
        s_queue.put(random_chunk)
        
        chunk_id += 1
    
def call_vertex():
    pass
