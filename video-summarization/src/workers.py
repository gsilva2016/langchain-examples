from datetime import datetime
import os
from queue import Queue
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

def send_summary_request(summary_q: Queue):
    while True:
        chunk_summaries = []
        
        while not summary_q.empty():
            chunk_summaries.append(summary_q.get())
        
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
            
            return merge_res
        else:
            print("Summary Merger: Waiting for chunk summaries to merge")
    
        time.sleep(25)
    
def ingest_frames_into_milvus(frame_q: Queue, milvus_manager: object):    
    while True:        
        # if not frame_q.empty():
        chunk = frame_q.get()
        
        if chunk is None:
            break
        
        print(f"Milvus: Ingesting {len(chunk['frames'])} chunk frames from {chunk['chunk_id']} into Milvus")
        try:
            response = milvus_manager.embed_img_and_store(chunk)
            
            print(f"Milvus: Chunk Frames Ingested into Milvus: {response['status']}, Total frames in chunk: {response['total_frames_in_chunk']}")
        
        except Exception as e:
            print(f"Milvus: Frame Ingestion Request failed: {e}")

def ingest_summaries_into_milvus(ingest_q: Queue, milvus_manager: object):
    while True:
        chunk_summaries = []
        
        while not ingest_q.empty():
            chunk_summaries.append(ingest_q.get())
        
        if chunk_summaries:
            print(f"Milvus: Ingesting {len(chunk_summaries)} chunk summaries into Milvus")
            try:
                response = milvus_manager.embed_txt_and_store(chunk_summaries)
                print(f"Milvus: Chunk Summaries Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
        
            except Exception as e:
                print(f"Milvus: Chunk Summaries Ingestion Request failed: {e}")
        else:
            print("Milvus: Waiting for chunk summaries to ingest")
    
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

def get_sampled_frames(chunk_queue: Queue, milvus_frames_queue: Queue, vlm_queue: Queue, max_num_frames: int = 64, resolution: list = [], save_frame: bool = False):
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
        chunk = chunk_queue.get()
        
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

def generate_chunk_summaries(vlm_q: Queue, milvus_summaries_queue: Queue, merger_queue: Queue):
    # To be replaced to logic to call miniCPM service
    while True:
        time.sleep(10)
        
        chunk = vlm_q.get()
        
        if chunk is None:
            break

        random_chunk = {
            "video_path": chunk["video_path"],
            "chunk_id": chunk["chunk_id"],
            "chunk_path": chunk["chunk_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_summary": f"Writing some random data here for chunk {chunk['chunk_id']}.",
        }
        
        print(f"Generating chunk summary for chunk {chunk['chunk_id']}")
        
        milvus_summaries_queue.put(random_chunk)
        merger_queue.put(random_chunk)
        
def generate_chunks(video_path: str, chunk_duration: int, chunk_overlap: int, chunk_queue: Queue, chunking_mechanism: str = "sliding_window"):
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
