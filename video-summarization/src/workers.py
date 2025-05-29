from datetime import datetime
import os
import queue
import time
import uuid
from dotenv import load_dotenv
from langchain_summarymerge_score import SummaryMergeScoreTool
from langchain_videochunk import VideoChunkLoader
import numpy as np
import requests
from PIL import Image
import time
from decord import VideoReader, cpu
from openvino import Tensor
from common.milvus.milvus_wrapper import MilvusManager

load_dotenv()
SUMMARY_MERGER_ENDPOINT = os.environ.get("SUMMARY_MERGER_ENDPOINT", None)

# This will be removed. This is a dummy placeholder to simulate chunk summaries.
ACTIVITIES = [
'''    **Overall Summary**
The video captures a sequence of moments inside a retail store, focusing on the checkout area and the surrounding aisles. The timestamp indicates the footage was taken on Tuesday, May 21, 2024, at 06:42:52.

**Activity Observed**
1. The video shows a cashier's station with a computer monitor and a cash drawer.
2. The aisles are stocked with various products, including snacks and beverages.
3. There is a visible customer interaction area where a customer is seen engaging with the cashier.
4. The floor is clean and well-maintained, with a green circular mat near the cashier's station.
5. The store appears to be open and operational with no visible signs of disturbance or damage.

**Potential Suspicious Activity**
1. No overt signs of shoplifting or suspicious behavior are observed in the provided frames. The customer interactions seem normal, and there is no evidence of theft or tampering with merchandise.

**Conclusion**
Based on the analysis, there is no evidence of shoplifting or suspicious activity within the observed frames. The store appears to be functioning normally with no immediate concerns.''',

'''
**Overall Summary**
The video captures a sequence of moments inside a retail store, focusing on the checkout area and the surrounding aisles. The footage is taken from a surveillance camera positioned above the store, providing a clear view of the activities below.

**Activity Observed**
1. The video shows a cashier at the checkout counter, attending to transactions.
2. Shelves stocked with various products are visible on both sides of the aisle.
3. The floor is clean and well-maintained, with a green circular mat near the checkout area.
4. There are no visible customers or staff in the immediate vicinity of the checkout counter.
5. The lighting in the store is bright, illuminating the entire area clearly.

**Potential Suspicious Activity**
1. There is no visible evidence of shoplifting or suspicious behavior in the provided frames. The cashier appears to be engaged in routine transactions, and there are no items being concealed or removed from view.

**Conclusion**
Based on the analysis, the video does not show any overt signs of shoplifting or suspicious activity. The store appears to be operating normally with standard retail operations being conducted.
''',

'''
**Overall Summary**
The video captures a sequence of moments inside a retail store, focusing on the checkout area and the surrounding aisles. The camera angle remains consistent throughout, providing a clear view of the store's layout and the activities within.

**Activity Observed**
1. The video shows a cashier at the checkout counter, handling transactions.
2. Shelves stocked with various products are visible on the right side of the frame.
3. The floor is clean and well-maintained, with a green circular mat near the checkout area.
4. There is a visible price tag on the left side of the frame, indicating the store's pricing information.
5. The lighting in the store is bright, illuminating the entire area clearly.

**Potential Suspicious Activity**
1. No overt signs of shoplifting or suspicious behavior are observed in the provided frames. The cashier appears to be engaged in routine transactions, and there are no individuals in the immediate vicinity of the checkout counter or the aisles that could indicate any illicit activity.

**Conclusion**
Based on the analysis of the video, no shoplifting or suspicious activities are detected within the observed frames. The environment appears orderly and the staff is performing their duties as expected.
'''
]

def send_summary_request(summary_q: queue.Queue):
    while True:
        chunk_summaries = []
        while not summary_q.empty():
            chunk = summary_q.get()
            if chunk is None:
                # exit the thread if None is received 
                return  
            chunk_summaries.append(chunk)
        
        if chunk_summaries:
            print(f"Summary Merger: Received {len(chunk_summaries)} chunk summaries for merging")
            
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
            
        else:
            print("Summary Merger: Waiting for chunk summaries to merge")
    
        time.sleep(25)
    
def ingest_frames_into_milvus(frame_q: queue.Queue, milvus_manager: object):    
    while True:        
        # if not frame_q.empty():
        try:
            chunk = frame_q.get()
            
            if chunk is None:
                break
            
        except queue.Empty:
            continue
        
        print(f"Milvus: Ingesting {len(chunk['frames'])} chunk frames from {chunk['chunk_id']} into Milvus")
        try:
            response = milvus_manager.embed_img_and_store(chunk)
            
            print(f"Milvus: Chunk Frames Ingested into Milvus: {response['status']}, Total frames in chunk: {response['total_frames_in_chunk']}")
        
        except Exception as e:
            print(f"Milvus: Frame Ingestion Request failed: {e}")

def ingest_summaries_into_milvus(ingest_q: queue.Queue, milvus_manager: object):
    while True:
        chunk_summaries = []
        
        try:
            chunk = ingest_q.get(timeout=1)
            chunk_summaries.append(chunk)
            
            if chunk is None:
                break
            
        except queue.Empty as e:
            continue
        
        print(f"Milvus: Ingesting {len(chunk_summaries)} chunk summaries into Milvus")
        try:
            response = milvus_manager.embed_txt_and_store(chunk_summaries)
            print(f"Milvus: Chunk Summaries Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
    
        except Exception as e:
            print(f"Milvus: Chunk Summaries Ingestion Request failed: {e}")

        time.sleep(25)

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

def get_sampled_frames(chunk_queue: queue.Queue, milvus_frames_queue: queue.Queue, vlm_queue: queue.Queue, max_num_frames: int = 64, resolution: list = [], save_frame: bool = False):
    # To be replaced with module from common/sampler for example    
    def uniform_sample(l: list, n: int) -> list:
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]
    
    def save_frames(frames: np.ndarray, frame_idx: list, video_path: str):
        os.makedirs(f"{video_path}", exist_ok=True)
        
        # Save the sampled frames as images
        for i, idx in enumerate(frame_idx):
            img = Image.fromarray(frames[i])
            img.save(f"{video_path}/frame_{idx}.jpg")
                
    while True:
        # if not input_queue.empty():
        try:
            chunk = chunk_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if chunk is None:
            break
        
        video_path = chunk["chunk_path"]
        src_video_path = chunk["video_path"]

        if len(resolution) != 0:
            vr = VideoReader(video_path, width=resolution[0],
                            height=resolution[1], ctx=cpu(0))
        else:
            vr = VideoReader(video_path, ctx=cpu(0))

        frame_idx = [i for i in range(0, len(vr), max(1, int(len(vr) / max_num_frames)))]
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        
        name = os.path.basename(video_path) + os.path.basename(src_video_path)
        
        if save_frame:
            save_frames(frames, frame_idx, name)

        # frames = [Tensor(v.astype('uint8')) for v in frames]
        print(f"Sampling frames from chunk: {chunk['chunk_id']} and Num frames sampled: {frames.shape[0]}")
        sampled = {
            "video_path": chunk["video_path"],
            "chunk_id": chunk["chunk_id"],
            "frames": frames,
            "frame_ids": frame_idx,
            "chunk_path": chunk["chunk_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"]
        }
        vlm_queue.put(sampled)
        milvus_frames_queue.put(sampled)
        
    print("Sampling completed")
    vlm_queue.put(None)
    milvus_frames_queue.put(None)

def generate_chunk_summaries(vlm_q: queue.Queue, milvus_summaries_queue: queue.Queue, merger_queue: queue.Queue):
    # To be replaced to logic to call miniCPM service
    
    while True:
        time.sleep(10)
        
        try:
            chunk = vlm_q.get(timeout=1)
        except queue.Empty:
            continue
        
        if chunk is None:
            break
        
        if "chunk_0" in chunk["chunk_path"]:
            random_chunk = {
                "video_path": chunk["video_path"],
                "chunk_id": chunk["chunk_id"],
                "chunk_path": chunk["chunk_path"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "chunk_summary": ACTIVITIES[0],
            }
        else:
            random_chunk = {
                "video_path": chunk["video_path"],
                "chunk_id": chunk["chunk_id"],
                "chunk_path": chunk["chunk_path"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "chunk_summary": ACTIVITIES[1],
            }
        
        print(f"Generating chunk summary for chunk {chunk['chunk_id']}")
        
        milvus_summaries_queue.put(random_chunk)
        merger_queue.put(random_chunk)
    
    milvus_summaries_queue.put(None)
    merger_queue.put(None)
        
def generate_chunks(video_path: str, chunk_duration: int, chunk_overlap: int, chunk_queue: queue.Queue, chunking_mechanism: str = "sliding_window"):
    loader = VideoChunkLoader(
        video_path=video_path,
        chunking_mechanism=chunking_mechanism,
        chunk_duration=chunk_duration,
        chunk_overlap=chunk_overlap)
    
    for doc in loader.lazy_load():
        print(f"Chunking video: {video_path} and chunk path: {doc.metadata['chunk_path']}")
        chunk = {
            "video_path": doc.metadata['source'],
            "chunk_id": f"{uuid.uuid4()}_{doc.metadata['chunk_id']}",
            "chunk_path": doc.metadata['chunk_path'],
            "chunk_metadata": doc.page_content,
            "start_time": doc.metadata['start_time'],
            "end_time": doc.metadata['end_time'],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }            
        chunk_queue.put(chunk)
    print(f"Chunk generation completed for {video_path}")
    
def call_vertex():
    pass
