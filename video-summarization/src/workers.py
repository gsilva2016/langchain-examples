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
import io
import base64

load_dotenv()
OVMS_ENDPOINT = os.environ.get("OVMS_ENDPOINT", None)
VLM_MODEL = os.environ.get("VLM_MODEL", "openbmb/MiniCPM-V-2_6")

def send_summary_request(summary_q: queue.Queue, n: int = 3):
    summary_merger = SummaryMergeScoreTool(
                    api_base=OVMS_ENDPOINT,
                )
    while True:
        chunk_summaries = []
        while len(chunk_summaries) < n:
            chunk = summary_q.get()  
            # exit the thread if None is received and wait for all current summaries to be processed
            if chunk is None:
                if chunk_summaries:
                    summary_q.put(None)  
                    break
                else:
                    return
            chunk_summaries.append(chunk)
        
        # get atleast n summaries to process, if there are more - process them all
        # MergeTool will handle them in batches 
        while True:
            try:
                chunk = summary_q.get_nowait()
                if chunk is None:
                    if chunk_summaries:
                        summary_q.put(None)
                        break
                    else:
                        return
                chunk_summaries.append(chunk)
            except queue.Empty:
                break

        if chunk_summaries:
            print(f"Summary Merger: Received {len(chunk_summaries)} chunk summaries for merging")
            formatted_req = {
                "summaries": {chunk["chunk_id"]: chunk["chunk_summary"] for chunk in chunk_summaries}
            }
            print(f"Summary Merger: Sending {len(chunk_summaries)} chunk summaries for merging")
            try:
                merge_res = summary_merger.invoke(formatted_req)
                print(f"Overall Summary: {merge_res['overall_summary']}")
                print(f"Anomaly Score: {merge_res['anomaly_score']}")
            except Exception as e:
                print(f"Summary Merger: Request failed: {e}")
        else:
            print("Summary Merger: Waiting for chunk summaries to merge")
    
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

def generate_chunk_summaries(vlm_q: queue.Queue, milvus_summaries_queue: queue.Queue, merger_queue: queue.Queue, 
                             prompt: str, max_new_tokens: int):
    
    while True:        
        try:
            chunk = vlm_q.get(timeout=1)

            if chunk is None:
                break

        except queue.Empty:
            print("VLM: No chunks available for summary generation")
            continue

        print(f"VLM: Generating chunk summary for chunk {chunk['chunk_id']}")

        content = [{"type": "text", "text": prompt}]
        for frame in chunk["frames"]:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            frame_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                            })

        data = {
            "model": VLM_MODEL,
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond in english."
                },
                {
                    "role": "user",
                    "content": content 
                }
            ]
        }

        response = requests.post(OVMS_ENDPOINT, 
                                 json=data, 
                                 headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            output_json = response.json()
            output_text = output_json["choices"][0]["message"]["content"]
            print("Response JSON:", output_json)
        else:
            print("Error:", response.status_code, response.text)
            continue
        
        chunk_summary = {
            "video_path": chunk["video_path"],
            "chunk_id": chunk["chunk_id"],
            "chunk_path": chunk["chunk_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_summary": output_text,
        }
        
        milvus_summaries_queue.put(chunk_summary)
        merger_queue.put(chunk_summary)

    print("VLM: Ending service")
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
