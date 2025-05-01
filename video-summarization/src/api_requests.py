import os
import time
from dotenv import load_dotenv
import numpy as np
import requests

load_dotenv()
SUMMARY_MERGER_ENDPOINT = os.environ.get("SUMMARY_MERGER_ENDPOINT", None)
MILVUS_INGESTION_ENDPOINT = os.environ.get("MILVUS_INGESTION_ENDPOINT", None)
MILVUS_SIM_SEARCH_ENDPOINT = os.environ.get("MILVUS_SIM_SEARCH_ENDPOINT", None)
MILVUS_QUERY_ENDPOINT = os.environ.get("MILVUS_QUERY_ENDPOINT", None)

def send_summary_request(summary_q):
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
                response = requests.post(url=SUMMARY_MERGER_ENDPOINT, json=formatted_req)
                if response.status_code != 200:
                    print(f"Summary Merger: Error: {response.status_code}, {response.content}")
                
                merge_res = response.json()
                print(f"Overall Summary: {merge_res['overall_summary']}")
                print(f"Anomaly Score: {merge_res['anomaly_score']}")

            except requests.exceptions.RequestException as e:
                print(f"Summary Merger: Request failed: {e}")
            
            # return response.content
        else:
            print("Summary Merger: Waiting for chunk summaries to merge")
    
        time.sleep(30)

def ingest_into_milvus(ingest_q):
    while True:
        chunk_summaries = []
        
        while not ingest_q.empty():
            chunk_summaries.append(ingest_q.get())
        
        if chunk_summaries:
            formatted_req = {
                "data": chunk_summaries

            }

            # print(formatted_req)
            print(f"Milvus: Ingesting {len(chunk_summaries)} chunk summaries into Milvus")
            try:
                response = requests.post(url=MILVUS_INGESTION_ENDPOINT, json=formatted_req)
                if response.status_code != 200:
                    print(f"Milvus: Error: {response.status_code}, {response.content}")
                
                milvus_res = response.json()
                print(f"Milvus: Chunk Summaries Ingested into Milvus: {milvus_res['status']}, Total chunks: {milvus_res['total_chunks']}")
            
            except requests.exceptions.RequestException as e:
                print(f"Milvus: Request failed: {e}")
        else:
            print("Milvus: Waiting for chunk summaries to ingest")
    
        time.sleep(30)

def search_in_milvus(query_text):
    try:
        response = requests.get(url=MILVUS_SIM_SEARCH_ENDPOINT, params={"query": query_text})
        print(response.content)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.content}")
            return None
        
        return response.content
    
    except requests.exceptions.RequestException as e:
        print(f"Search in Milvus: Request failed: {e}")
        return None

def query_vectors(expr):
    try:
        response = requests.get(url=MILVUS_QUERY_ENDPOINT, params={"expr": expr})
        # print(response.content)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.content}")
            return None
        
        return response.content
    
    except requests.exceptions.RequestException as e:
        print(f"Query Vectors: Request failed: {e}")
        return None
    
def generate_chunk_summaries(i_queue, s_queue, f_queue):
    # To be replaced to logic to call miniCPM service
    chunk_id = 1
    while True:
        time.sleep(15)
        
        # Create some random frame data
        frame = np.random.rand(1, 3, 270, 480)
        
        random_chunk = {
            "camera_id": chunk_id,
            "chunk_id": f"chunk_camera_{chunk_id}_{chunk_id}",
            "chunk_path": f"video_chunks/cam_{chunk_id}/chunk_{chunk_id}.mp4",
            "chunk_summary": f"Writing some random data here for chunk {chunk_id}.",
            "start_time": f"random time",
            "end_time": f"random time"
        }
        print(f"Generating chunk summary for chunk {chunk_id}")
        
        f_queue.put(frame)
        i_queue.put(random_chunk)
        s_queue.put(random_chunk)
        
        chunk_id += 1
    
def call_vertex():
    pass
