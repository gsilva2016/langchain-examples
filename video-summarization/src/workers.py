from datetime import datetime
import os
import queue
import time
from dotenv import load_dotenv
from langchain_summarymerge_score import SummaryMergeScoreTool
from common.rtsploader.rtsploader_wrapper import RTSPChunkLoader
from common.sampler.frame_sampler import FrameSampler

import requests
from PIL import Image
import time
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

def get_sampled_frames(chunk_queue: queue.Queue, milvus_frames_queue: queue.Queue, vlm_queue: queue.Queue,
                       max_num_frames: int = 64, resolution: list = [], save_frame: bool = False):
    
    sampler = FrameSampler(max_num_frames=max_num_frames, resolution=resolution, save_frame=save_frame)

    while True:
        try:
            chunk = chunk_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if chunk is None:
            break
        
        # Sample frames from the video chunk
        print(f"SAMPLER: Sampling frames {max_num_frames} from chunk: {chunk['chunk_id']}")
        video_path = chunk["chunk_path"]
        try:
            frames_dict = sampler.sample_frames_from_video(video_path, chunk["detected_objects"])
        except Exception as e:
            print(f"SAMPLER: sampling failed: {e}")
        
        sampled = {
            "video_path": chunk["video_path"],
            "chunk_id": chunk["chunk_id"],
            "frames": frames_dict["frames"],
            "frame_ids": frames_dict["frame_ids"],
            "chunk_path": chunk["chunk_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
        }
        vlm_queue.put({**sampled, "detected_objects": frames_dict["detected_objects"]})
        milvus_frames_queue.put({**sampled, "detected_objects": chunk["detected_objects"]})
        
    print("SAMPLER: Sampling completed")
    vlm_queue.put(None)
    milvus_frames_queue.put(None)

def generate_chunk_summaries(vlm_q: queue.Queue, milvus_summaries_queue: queue.Queue, merger_queue: queue.Queue, 
                             prompt: str, max_new_tokens: int, obj_detect_enabled: bool):
    
    while True:        
        try:
            chunk = vlm_q.get(timeout=1)
            if chunk is None:
                break

        except queue.Empty:
            continue
        print(f"VLM: Generating chunk summary for chunk {chunk['chunk_id']}")

        # Prepare the text prompt content for the VLM request
        content = [{"type": "text", "text": prompt}]
        
        # Prepare the frames for the VLM request
        for frame in chunk["frames"]:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            frame_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                            })
        
        # Add the object detection metadata to the VLM request
        if obj_detect_enabled:
            detected_objects = chunk["detected_objects"]
            
            # Format detected objects for LLM input
            detection_lines = []
            for d in detected_objects:
                frame_num = d.get('frame')
                objs = d.get('objects', [])
                if objs:
                    obj_descriptions = []
                    for obj in objs:
                        label = obj.get('label')
                        bbox = obj.get('bbox')
                        bbox_str = f"[{', '.join([f'{v:.2f}' for v in bbox])}]" if bbox else "[]"
                        obj_descriptions.append(f"{label} at {bbox_str}")
                    detection_lines.append(f"Frame {frame_num}: " + "; ".join(obj_descriptions))
            detection_text = (
                "Detected objects per frame:\n" +
                "\n".join(detection_lines) +
                "\nPlease use this information in your analysis."
            )
            content.append({"type": "text", "text": detection_text}) 

        # Package all request data for the VLM
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

        # Send the request to the VLM model endpoint
        response = requests.post(OVMS_ENDPOINT, 
                                 json=data, 
                                 headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            output_json = response.json()
            output_text = output_json["choices"][0]["message"]["content"]
            print("VLM: Model response:", output_json)
        else:
            print("VLM: Error:", response.status_code, response.text)
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
        
def generate_chunks(video_path: str, chunk_duration: int, chunk_overlap: int, chunk_queue: queue.Queue,
                    obj_detect_enabled: bool, obj_detect_path: str, obj_detect_sample_rate: int, 
                    obj_detect_threshold: float, chunking_mechanism: str = "sliding_window"):

    # Initialize the video chunk loader
    chunk_args = {
        "window_size": chunk_duration,
        "overlap": chunk_overlap,
    }
    if obj_detect_enabled:
        chunk_args.update({
            "obj_detect_enabled": obj_detect_enabled,
            "dfine_path": obj_detect_path,
            "dfine_sample_rate": obj_detect_sample_rate,
            "detection_threshold": obj_detect_threshold
        })

    loader = RTSPChunkLoader(
        rtsp_url=video_path,
        chunk_type=chunking_mechanism,
        chunk_args=chunk_args,
    )
    
    # Generate chunks
    for doc in loader.lazy_load():
        print(f"CHUNK LOADER: Chunking video: {video_path} and chunk path: {doc.metadata['chunk_path']}")
        chunk = {
            "video_path": doc.metadata['source'],
            "chunk_id": doc.metadata['chunk_id'],
            "chunk_path": doc.metadata['chunk_path'],
            "chunk_metadata": doc.page_content,
            "start_time": doc.metadata['start_time'],
            "end_time": doc.metadata['end_time'],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detected_objects": doc.metadata["detected_objects"]
        }
        chunk_queue.put(chunk)
    print(f"CHUNK LOADER: Chunk generation completed for {video_path}")
    
def call_vertex():
    pass
