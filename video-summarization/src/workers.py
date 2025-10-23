from collections import defaultdict
from datetime import datetime, timedelta
import random
import threading
import uuid
import cv2
import numpy as np
from common.tracker.person_detection import PersonDetector
from common.tracker.tracking import ReIDExtractor
from common.tracker.deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from common.tracker.deepsort_utils.tracker import Tracker
from common.tracker.deepsort_utils.detection import Detection, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy
from common.tracker.tracking import draw_boxes
import os
import queue
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from langchain_summarymerge_score import SummaryMergeScoreTool
from common.rtsploader.rtsploader_wrapper import RTSPChunkLoader
from common.sampler.frame_sampler import FrameSampler

from common.milvus.milvus_wrapper import MilvusManager
from langchain_openvino_multimodal import OpenVINOBlipEmbeddings

import requests
from PIL import Image
import io
import base64
import time

load_dotenv()

# Thresholds
DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", 0.8))
OVMS_ENDPOINT = os.environ.get("OVMS_ENDPOINT", None)
VLM_MODEL = os.environ.get("VLM_MODEL", "openbmb/MiniCPM-V-2_6")
SIM_SCORE_THRESHOLD = float(os.environ.get("REID_SIM_SCORE_THRESHOLD", 0.65))
TOO_SIMILAR_THRESHOLD = float(os.environ.get("TOO_SIMILAR_THRESHOLD", 0.95))
AMBIGUITY_MARGIN = float(os.environ.get("AMBIGUITY_MARGIN", 0.15))
PARTITION_CREATION_INTERVAL = int(os.environ.get("PARTITION_CREATION_INTERVAL", 1))
TRACKING_LOGS_GENERATION_TIME_SECS = float(os.environ.get("TRACKING_LOGS_GENERATION_TIME_SECS", 1.0))
MAX_EVENTS_BATCH = int(os.environ.get("MAX_EVENTS_BATCH", 10))

global_track_locks = defaultdict(threading.Lock)
global_assignment_lock = threading.Lock()

# Shared across threads
track_rolling_avgs = defaultdict(lambda: deque(maxlen=8))
global_mean = {}

# Locks for concurrency
local_state_lock = threading.Lock()
global_mean_lock = threading.Lock()

        
def send_summary_request(summary_q: queue.Queue, n: int = 3):
    summary_merger = SummaryMergeScoreTool(api_base=OVMS_ENDPOINT)

    summaries = []
    last = False

    while True:
        while len(summaries) < n and not last:
            chunk = summary_q.get()
            if chunk is None:
                last = True
                break
            summaries.append(chunk)

        while True:
            try:
                chunk = summary_q.get_nowait()
                if chunk is None:
                    last = True
                    break
                summaries.append(chunk)
            except queue.Empty:
                break

        if len(summaries) >= n or (last and summaries):
            print(f"[Summary Merger]: Received {len(summaries)} chunk summaries for merging")
            formatted_req = {
                "summaries": {chunk["chunk_id"]: chunk["chunk_summary"] for chunk in summaries}
            }
            print(f"[Summary Merger]: Sending {len(summaries)} chunk summaries for merging")
            try:
                merge_res = summary_merger.invoke(formatted_req)
                print(f"[Summary Merger]: Overall Summary: {merge_res['overall_summary']}")
                print(f"[Summary Merger]: Anomaly Score: {merge_res['anomaly_score']}")
            except Exception as e:
                print(f"[Summary Merger]: Request failed: {e}")

            summaries = []

        if last and not summaries:
            print("[Summary Merger]: All summaries processed, exiting.")
            return
    
def ingest_summaries_into_milvus(milvus_summaries_q: queue.Queue, milvus_manager: MilvusManager, ov_blip_embedder: OpenVINOBlipEmbeddings):    
    summaries = []
    last = False

    while True:
        try:
            chunk = milvus_summaries_q.get(timeout=1)
            if chunk is None:
                last = True
            else:
                summaries.append(chunk)
        except queue.Empty:
            pass  

        if summaries and (last or milvus_summaries_q.empty()):
            print(f"[Milvus]: Ingesting {len(summaries)} chunk summaries into Milvus")
            try:
                all_summaries = [item["chunk_summary"] for item in summaries]
                embeddings = ov_blip_embedder.embed_documents(all_summaries)
                print(f"[Milvus]: Generated {len(embeddings)} text embeddings of Shape: {embeddings[0].shape}")

                metadatas = [
                    {
                        "video_path": item["video_path"],
                        "chunk_id": item["chunk_id"],
                        "chunk_path": item["chunk_path"],
                        "start_time": item["start_time"],
                        "end_time": item["end_time"],
                        "detected_objects": item.get("detected_objects", []),
                        "mode": "text",
                        "summary": item["chunk_summary"],
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    } for item in summaries
                ]

                response = milvus_manager.insert_data(collection_name=os.environ.get("VIDEO_COLLECTION_NAME", "video_chunks"),
                                                      vectors=embeddings,
                                                      metadatas=metadatas)

                print(f"[Milvus]: Chunk Summaries Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
            except Exception as e:
                print(f"[Milvus]: Chunk Summaries Ingestion Request failed: {e}")
            summaries = []  

        if last and not summaries:
            print("[Milvus]: All summaries ingested, exiting.")
            break

def ingest_frames_into_milvus(frame_q: queue.Queue, milvus_manager: MilvusManager, ov_blip_embedder: OpenVINOBlipEmbeddings):    
    while True:        
        # if not frame_q.empty():
        try:
            chunk = frame_q.get()
            
            if chunk is None:
                break
            
        except queue.Empty:
            continue

        print(f"[Milvus]: Ingesting {len(chunk['frames'])} chunk frames from {chunk['chunk_id']} into Milvus")
        try:
            all_sampled_images = chunk["frames"]
            embeddings = ov_blip_embedder.embed_images(all_sampled_images)
            print(f"Generated {len(embeddings)} img embeddings of Shape: {embeddings[0].shape}")
            
            metadatas = []
            for idx in chunk["frame_ids"]:
                objects = []
                for x in chunk.get("detected_objects", []):
                    if x.get("frame") == idx:
                        objects = x.get("objects", [])
                        break
                metadatas.append({
                    "video_path": chunk["video_path"],
                    "chunk_id": chunk["chunk_id"],
                    "frame_id": idx,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["start_time"],
                    "chunk_path": chunk["chunk_path"],
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "mode": "image",
                    "detected_objects": str(objects)
                })
                            
            response = milvus_manager.insert_data(collection_name=os.environ.get("VIDEO_COLLECTION_NAME", "video_chunks"),
                                        vectors=embeddings,
                                        metadatas=metadatas)

            print(f"[Milvus]: Chunk Frames Ingested into Milvus: {response['status']}, Total frames in chunk: {response['total_chunks']}")

        except Exception as e:
            print(f"[Milvus]: Frame Ingestion Request failed: {e}")

def get_sampled_frames(chunk_queue: queue.Queue, milvus_frames_queue: queue.Queue, vlm_queue: queue.Queue,
                       max_num_frames: int = 32, resolution: list = [], save_frame: bool = False):
    
    sampler = FrameSampler(max_num_frames=max_num_frames, resolution=resolution, save_frame=save_frame)

    while True:
        try:
            chunk = chunk_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if chunk is None:
            break
        
        # Sample frames from the video chunk
        print(f"[SAMPLER]: Sampling frames {max_num_frames} from chunk: {chunk['chunk_id']}")
        video_path = chunk["chunk_path"]
        try:
            frames_dict = sampler.sample_frames_from_video(video_path, chunk["detected_objects"])
        except Exception as e:
            print(f"[SAMPLER]: sampling failed: {e}")
        
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

    print("[SAMPLER]: Sampling completed")
    vlm_queue.put(None)
    milvus_frames_queue.put(None)

def generate_chunk_summaries(vlm_q: queue.Queue, milvus_summaries_queue: queue.Queue, merger_queue: queue.Queue, 
                             prompt: str, max_new_tokens: int, obj_detect_enabled: bool, 
                             milvus_manager: MilvusManager = None, tracking_enabled: bool = False):
    
    while True:        
        try:
            chunk = vlm_q.get(timeout=1)
            if chunk is None:
                break

        except queue.Empty:
            continue
        print(f"[VLM]: Generating chunk summary for chunk {chunk['chunk_id']}")
        
        # Prepare the frames for the VLM request
        content = []
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
            
            # Format detected objects for VLM input
            detection_lines = []
            for d in detected_objects:
                frame_num = d.get("frame")
                objs = d.get("objects", [])
                if objs:
                    obj_descriptions = []
                    for obj in objs:
                        label = obj.get("label")
                        bbox = obj.get("bbox")
                        bbox_str = f"[{', '.join([f'{v:.2f}' for v in bbox])}]" if bbox else "[]"
                        obj_descriptions.append(f"{label} at {bbox_str}")
                    detection_lines.append(f"Frame {frame_num}: " + "; ".join(obj_descriptions))
            detection_text = (
                "Detected objects per frame:\n" +
                "\n".join(detection_lines) +
                "\nPlease use this information in your analysis."
            )
            content.append({"type": "text", "text": detection_text})
        
        # Add tracking information to enrich VLM context
        # AS AN EXAMPLE:
        # Current tracking context - 12 recent detections:
        # Active tracked persons: 3 unique individuals
        # Person 123abc: last seen at [100.0, 200.0, 150.0, 250.0] in chunk_001
        # Person 456def: last seen at [300.0, 400.0, 350.0, 450.0] in chunk_001
        # Consider this tracking context in your summary.
        if tracking_enabled and milvus_manager:
            print("[VLM]: adding tracking info")
            try:
                # Get current tracking info
                tracking_info = get_tracking_info(
                    milvus_manager,
                    chunk_start_time=chunk["start_time"],
                    video_path=chunk["chunk_path"]
                )

                print("[VLM]:", tracking_info)
                if tracking_info["active_global_ids"]:
                    tracking_lines = [
                        f"Current tracking context - {tracking_info['total_detections']} recent detections:",
                        f"Active tracked persons: {len(tracking_info['active_global_ids'])} unique individuals"
                    ]
                    
                    # Add specific track details
                    for track_id in list(tracking_info["active_global_ids"]):
                        source_info = tracking_info["track_sources"].get(track_id, {})
                        bbox = tracking_info["track_locations"].get(track_id, [])
                        if bbox:
                            bbox_str = f"[{', '.join([f'{v:.1f}' for v in bbox])}]"
                            tracking_lines.append(f"Person {track_id}: last seen at {bbox_str} in {source_info.get('chunk_id', 'unknown')}")
                    
                    tracking_text = ("\n".join(tracking_lines) +
                        "\nConsider this tracking context in your summary. If specifying a person, please denote them by their global id.")

                    print("[VLM]:", tracking_text)
                    content.append({"type": "text", "text": tracking_text})
            except Exception as e:
                print(f"[VLM]: Failed to get tracking info: {e}")
                
        # Prepare the text prompt content for the VLM request
        content.append({"type": "text", "text": prompt})

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
            print("[VLM]: Model response:", output_json)
        else:
            print("[VLM]: Error:", response.status_code, response.text)
            continue
        
        chunk_summary = {
            "video_path": chunk["video_path"],
            "chunk_id": chunk["chunk_id"],
            "chunk_path": chunk["chunk_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_summary": output_text,
            "detected_objects": chunk["detected_objects"]
        }
        
        milvus_summaries_queue.put(chunk_summary)
        merger_queue.put(chunk_summary)

    print("[VLM]: Ending service")
    milvus_summaries_queue.put(None)
    merger_queue.put(None)
        
def generate_chunks(video_path: str, chunk_duration: int, chunk_overlap: int, chunk_queue: queue.Queue,
                    tracking_chunk_queue: queue.Queue, obj_detect_enabled: bool, obj_detect_path: str, 
                    obj_detect_sample_rate: int, obj_detect_threshold: float, 
                    chunking_mechanism: str = "sliding_window"):
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
        chunk_args=chunk_args
    )
    
    # Generate chunks
    for doc in loader.lazy_load():
        print(f"[CHUNK LOADER]: Chunking video: {video_path} and chunk path: {doc.metadata['chunk_path']}")
        chunk = {
            "video_path": doc.metadata["source"],
            "chunk_id": doc.metadata["chunk_id"],
            "chunk_path": doc.metadata["chunk_path"],
            "chunk_metadata": doc.page_content,
            "start_time": doc.metadata["start_time"],
            "end_time": doc.metadata["end_time"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detected_objects": doc.metadata["detected_objects"]
        }
        chunk_queue.put(chunk)
        if tracking_chunk_queue is not None:
            print("[CHUNK LOADER]: placing in tracking queue")
            tracking_chunk_queue.put(chunk)
    print(f"[CHUNK LOADER]: Chunk generation completed for {video_path}")

def generate_deepsort_tracks(tracking_chunk_queue: queue.Queue, tracking_results_queue: queue.Queue,
                             det_model_path: str, reid_model_path: str, device: str = "AUTO",
                             nn_budget: int = 100, max_cosine_distance: float = 0.5, metric_type: str = "cosine",
                             max_iou_distance: float = 0.7, max_age: int = 100, n_init: int = 1,
                             resize_dim: tuple = (700, 450), sampling_rate: int = 1, det_thresh: float = 0.5,
                             write_video: bool = True):
    """
    Chunk level DeepSORT tracking. Processes video chunks from tracking_chunk_queue,
    runs detection and tracking, and puts results in tracking_results_queue.
    Optionally writes output video with tracks if write_video is True.
    """
    # Initialize the person detector, re-identification extractor, metric, and tracker
    detector = PersonDetector(det_model_path, device=device, thresh=det_thresh)
    extractor = ReIDExtractor(reid_model_path, device=device)
    metric = NearestNeighborDistanceMetric(metric_type, max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
    sampler = None
    batch_process = 50

    while True:
        start_t = time.time()
                
        # Get a chunk from the tracking queue
        try:
            chunk = tracking_chunk_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # End if none
        if chunk is None:
            break

        # Grab the video chunk file
        video_path = chunk["chunk_path"]

        # Initialte sampler if not already done
        if sampler is None:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            
            fps = vr.get_avg_fps() if hasattr(vr, 'get_avg_fps') else 30
            # sample_interval = max(1, int(fps // reid_sampling_rate))
            # print(f"--------------DeepSORT: Video FPS: {fps}, Sampling every {sample_interval} frames for ReID with rate {reid_sampling_rate}fps")
            
            if sampling_rate <= 1:
                max_num_frames = int(len(vr))  # sample every frame
            else:
                max_num_frames = max(1, int(len(vr) / sampling_rate))  # sample every Nth frame
            sampler = FrameSampler(max_num_frames=max_num_frames, resolution=list(resize_dim), save_frame=False)
        print(f"[DeepSORT]: Processing {max_num_frames} frames from {video_path}")

        # Sample frames from the video chunk
        sampled = sampler.sample_frames_from_video(video_path, [])
        frames = sampled["frames"]
        frame_ids = sampled["frame_ids"]

        # Setup video writer for output video with tracks if enabled
        if write_video:                
            output_video_path = os.path.splitext(video_path)[0] + "_tracks.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if len(frames) > 0:
                h, w = frames[0].shape[:2]
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            else:
                out = None
        else:
            out = None

        # Run tracking on sampled frames
        h, w = None, None
        chunk_tracking_results = []
        for (i, frame), frame_id in zip(enumerate(frames), frame_ids):
            # Preprocess frame for detection
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            h, w = frame.shape[:2]
            input_image = detector.preprocess(frame)

            # Run person detection model
            output = detector.compiled_model(input_image)[detector.output_layer]
            bbox_xywh, score, label = detector.process_results(h, w, results=output)

            # Crop detected person regions for re-identification
            img_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                img = frame[y1:y2, x1:x2]
                img_crops.append(img)

            # Extract re-identification features for each crop
            if img_crops:
                img_batch = extractor.batch_preprocess(img_crops)
                features = extractor.predict(img_batch)
            else:
                features = np.array([])

            # Convert bounding boxes to tracker format
            bbox_tlwh = xywh_to_tlwh(bbox_xywh)

            # Create Detection objects for tracker
            detections = [Detection(bbox_tlwh[i], features[i]) for i in range(features.shape[0])]

            # Predict and update tracker state
            tracker.predict()
            tracker.update(detections)

            # Get features for confirmed tracks and align with outputs by track_id
            track_features_dict = tracker.get_track_features()

            # Gather track info
            outputs = []
            reid_features = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
                reid_features.append(track_features_dict.get(track_id, []))

            # Prepare output arrays for bounding boxes and identities
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
            else:
                bbox_xyxy = np.array([])
                identities = np.array([])

            # Draw tracks on frame and write to output video if enabled
            if out is not None:
                frame_with_tracks = draw_boxes(frame.copy(), bbox_xyxy, identities)
                out.write(frame_with_tracks)
            
            if identities is None or len(identities) == 0:
                continue
            
            # Accumulate tracking results for this frame
            tracking_res = {
                "frame_id": frame_id,
                "chunk_id": chunk["chunk_id"],
                "bboxes": bbox_xyxy.tolist(),
                "object_class": ["person"] * len(bbox_xyxy),  
                "track_ids": [int(trackid) for trackid in identities],
                "reid_embeddings": [
                    embedding.tolist()
                    for feature_list in reid_features
                    for embedding in feature_list
                ],
                "video_path": chunk["video_path"],
                "chunk_path": chunk["chunk_path"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"]
            }
        
            chunk_tracking_results.append(tracking_res)
        
            # print("length of chunk tracking results:", len(chunk_tracking_results)) 
            if len(chunk_tracking_results) >= batch_process:
                tracking_results_queue.put(chunk_tracking_results)
                chunk_tracking_results = []

        if len(chunk_tracking_results) > 0:
            tracking_results_queue.put(chunk_tracking_results)
            
        # Release video writer if enabled
        if out is not None:
            # print(f"[DeepSORT]: Output video with tracks written to {output_video_path}")
            out.release()

        end_t = time.time() - start_t
        print(f"[DeepSORT]: ******** Finished processing {video_path}: {len(frames)} frames in {end_t:.2f}s")

    print("[DeepSORT]: Tracking completed")
    tracking_results_queue.put(None)

def show_tracking_data(global_track_ids: dict, interval = 5):
    try:        
        while True:
            time.sleep(interval)
            print("--------------------------Current state of global track IDs:------------------------------------------")
            
            if -1 in global_track_ids:
                print("End sentinel found in global table. Ending.")
                break

            print(f"Total active tracks: {len(global_track_ids)}")
            for track_id, info in global_track_ids.items():
                print(f"Track ID: {track_id},       Info: {info}")
                print("****************************")
            print("-----------------------------------------------------------------------------------------------------")
    except Exception as e:
        print(f"[Track Viewer]: Exception: {e}")
    finally:
        print("[Track Viewer]: Stopping track viewer.")
        print(f"Total active tracks: {len(global_track_ids) - 1}")
        for track_id, info in global_track_ids.items():
            print(f"Track ID: {track_id},       Info: {info}")
            print("****************************")
        print("-----------------------------------------------------------------------------------------------------")


def process_tracking_logs(tracking_logs_q: queue.Queue, milvus_manager: MilvusManager, ov_blip_embedder: OpenVINOBlipEmbeddings, collection_name: str = "tracking_logs"):
    last_event_time = defaultdict(lambda: 0)
    events = []
    wait_interval = 0.5  
    last_flush = time.time()

    while True:
        try:
            event = tracking_logs_q.get(timeout=wait_interval)
            if event is None:
                break

            gid = event["global_track_id"]
            now = time.time()

            # Process event only if greater than TRACKING_LOGS_GENERATION_TIME_SECS, prevents too many events
            if now - last_event_time[gid] >= TRACKING_LOGS_GENERATION_TIME_SECS:
                last_event_time[gid] = now
                events.append(event)

        except queue.Empty:
            pass

        # Now insert into Milvus if wait_interval has passed or we have atleast max_events
        if events and (len(events) >= MAX_EVENTS_BATCH or time.time() - last_flush >= wait_interval):
            try:
                texts = [e["description"] for e in events]
                embeddings = ov_blip_embedder.embed_documents(texts)
                milvus_manager.insert_data(
                    collection_name=collection_name,
                    vectors=embeddings,
                    metadatas=events
                )
                # print(f"[TrackingLogs]: Batch inserted {len(events)} events to Milvus")
            except Exception as e:
                print(f"[TrackingLogs]: Failed insert: {e}")
            finally:
                events = []
                last_flush = time.time()
            
def extract_frame_from_video(chunk_path, frame_id):
    # Use openCV to extract the specific frame
    cap = cv2.VideoCapture(chunk_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_id} from {chunk_path}")
    return frame

def get_tracking_info(milvus_manager: MilvusManager, chunk_start_time: str, 
                      video_path: str, collection_name: str = "reid_data") -> dict:
    """
    Probe Milvus DB for tracking information to enrich VLM context
    """
    # Use chunk_start_time for filtering collection
    # filter_expr = (
    #     f'metadata["timestamp"] >= "{chunk_start_time}" and '
    #     f'metadata["mode"] == "{collection_name}" and '
    #     f'metadata["video_path"] == "{video_path}"'
    # )
    filter_expr = f'metadata["chunk_path"] == "{video_path}"'
    
    results = milvus_manager.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["metadata"])
    
    # Process results into structured tracking dictionary
    tracking_summary = {
        "active_global_ids": set(),
        "track_locations": {},
        "track_sources": {},
        "total_detections": len(results) if isinstance(results, list) else 0
    }
    
    if isinstance(results, list):
        for result in results:
            metadata = result.get("metadata", {})
            global_id = metadata.get("global_track_id")
            if global_id:
                tracking_summary["active_global_ids"].add(global_id)
                tracking_summary["track_locations"][global_id] = metadata.get("bbox", [])
                tracking_summary["track_sources"][global_id] = {
                    "video_path": metadata.get("video_path"),
                    "chunk_id": metadata.get("chunk_id"),
                    "frame_id": metadata.get("frame_id"),
                    "timestamp": metadata.get("timestamp")
                }
    
    return tracking_summary

def visualize_tracking_data(visualization_queue: queue.Queue, tracker_dim: tuple = (700, 450)):
    writers = {}
    caps = {}
    try:
        while True:
            items = visualization_queue.get()
            if items is None:
                break

            for item in items:
                frame_data, local_to_global = item
                chunk_path = frame_data["chunk_path"]
                frame_id = frame_data["frame_id"]

                cap = caps.get(chunk_path)
                if cap is None:
                    cap = cv2.VideoCapture(chunk_path)
                    caps[chunk_path] = cap

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                # resize frame if needed
                ret, frame = cap.read()
                frame = cv2.resize(frame, tracker_dim)
                if not ret:
                    print(f"[Visualizer]: Failed to read frame {frame_id} from {chunk_path}")
                    continue

                video_src = chunk_path.split(".")[0]
                writer = writers.get(video_src)
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = f"{video_src}_reid_viz.mp4"
                    writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
                    writers[video_src] = (writer, out_path)

                writer_obj, out_path = writers[video_src]
                
                uniq_ppl = set()
                for bbox, local_id in zip(frame_data["bboxes"], frame_data["track_ids"]):
                    global_id = local_to_global.get(local_id, local_id)                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put text at the center of the box
                    cv2.putText(frame, f"GID {global_id}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"F {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    uniq_ppl.add(global_id)
                cv2.putText(frame, f"Unique IDs: {len(uniq_ppl)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                writer_obj.write(frame)
    finally:
        # Cleanup all resources
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        for writer_obj, out_path in writers.values():
            try:
                writer_obj.release()
                # print(f"[Visualizer] Finished {out_path}")
            except Exception:
                pass

def insert_reid_embeddings(frame: dict, milvus_manager: MilvusManager, collection_name: str = "reid_data"):
    """
    Insert ReID embeddings into Milvus with global ID assignment and rolling aggregation.
    """
    batch_embeddings, batch_metadatas = [], []
    global_assigned_ids, local_track_ids, is_new_tracks, global_track_sources = [], [], [], []

    identities = frame.get("track_ids", [])
    reid_embeddings = frame.get("reid_embeddings", [])
    frame_id = frame.get("frame_id", -1)

    now = datetime.now()
    partition_current = f"{collection_name}_{now.strftime('%Y%m%d_%H')}"
    partition_prev = f"{collection_name}_{(now - timedelta(hours=PARTITION_CREATION_INTERVAL)).strftime('%Y%m%d_%H')}"
    search_partitions = [partition_current, partition_prev]

    # Search outside lock for concurrency
    search_results_batch = milvus_manager.search(
        collection_name=collection_name,
        query_vector=reid_embeddings,
        partition_names=search_partitions
    )

    with global_assignment_lock:
        for i, emb in enumerate(reid_embeddings):
            emb = np.array(emb)
            search_results = search_results_batch[i] if i < len(search_results_batch) else []
            global_track_id, is_new_track, should_store = None, True, True
            sim_score = None

            # Parse search results
            if search_results:
                hit = search_results[0]
                sim_score = hit["distance"]
                metadata = hit["entity"]["metadata"]

                if sim_score > SIM_SCORE_THRESHOLD:
                    global_track_id = metadata.get("global_track_id")
                    is_new_track = False
                    if sim_score > TOO_SIMILAR_THRESHOLD:
                        should_store = False  # Too similar, unnecessary to store

                if SIM_SCORE_THRESHOLD - AMBIGUITY_MARGIN <= sim_score < SIM_SCORE_THRESHOLD:
                    continue

            # Assigning new GID here
            if not global_track_id:
                global_track_id = f"{uuid.uuid4().hex}_person"

            # Aggregating embeddings for the track here
            with local_state_lock:
                track_rolling_avgs[global_track_id].append(emb)
                avg_emb = np.mean(track_rolling_avgs[global_track_id], axis=0)

            with global_track_locks[global_track_id]:
                with global_mean_lock:
                    last_mean = global_mean.get(global_track_id)
                    drift = None
                    
                    if last_mean is not None:
                        drift = cosine_similarity(avg_emb.reshape(1, -1), last_mean.reshape(1, -1))[0, 0]
                        if drift < DIVERGENCE_THRESHOLD:
                            should_store = True
                            global_mean[global_track_id] = avg_emb
                        else:
                            should_store = False
                    else:
                        should_store = True
                        global_mean[global_track_id] = avg_emb
            
            # Store embedding and metadata if needed
            if should_store:
                metadata = {
                    "local_track_id": identities[i] if i < len(identities) else -1,
                    "global_track_id": global_track_id,
                    "video_path": frame["video_path"],
                    "chunk_id": frame["chunk_id"],
                    "chunk_path": frame["chunk_path"],
                    "start_time": frame["start_time"],
                    "end_time": frame["end_time"],
                    "mode": "reid",
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "bbox": frame["bboxes"][i] if i < len(frame["bboxes"]) else [],
                    "object_class": frame["object_class"][i] if i < len(frame["object_class"]) else None,
                    "frame_id": frame_id
                }
                batch_embeddings.append(avg_emb.tolist())
                batch_metadatas.append(metadata)

            local_track_ids.append(identities[i] if i < len(identities) else -1)
            global_assigned_ids.append(global_track_id)
            is_new_tracks.append(is_new_track)
            global_track_sources.append(f"{frame['video_path']}:{frame['chunk_path']}")

        # Insert batch if we have any
        if batch_embeddings:
            milvus_manager.insert_data(
                collection_name=collection_name,
                vectors=batch_embeddings,
                metadatas=batch_metadatas,
                partition_name=partition_current)

    return global_assigned_ids, local_track_ids, is_new_tracks, global_track_sources

def process_reid_embeddings(tracking_results_queue: queue.Queue, tracking_logs_q: queue.Queue, visualization_queue: queue.Queue, global_track_table: dict, milvus_manager: MilvusManager, collection_name: str = "reid_data"):
    """
    Processes tracking results and inserts ReID embeddings into Milvus with global ID assignment.
    Also updates global track table and sends data for visualization.
    """
    while True:
        try:
            frame_batch = tracking_results_queue.get(timeout=1)
            if frame_batch is None:
                with global_assignment_lock:
                    flush_embeddings, flush_metadatas = [], []
                    for gid, mean in list(global_mean.items()):
                        flush_embeddings.append(mean.tolist())
                        flush_metadatas.append({
                            "global_track_id": gid,
                            "mode": "reid",
                            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "event": "final_flush"
                        })
                    if flush_embeddings:
                        milvus_manager.insert_data(collection_name=collection_name, vectors=flush_embeddings, metadatas=flush_metadatas)
                
                visualization_queue.put(None)
                break
        except queue.Empty:
            continue

        viz_batch = []
        for frame in frame_batch:
            global_assigned_ids, local_track_ids, is_new_tracks, global_track_sources = insert_reid_embeddings(frame, milvus_manager, collection_name)
          
            for idx, global_track_id in enumerate(global_assigned_ids):
                with global_track_locks[global_track_id]:
                    if global_track_id not in global_track_table:
                        global_track_table[global_track_id] = {
                            "is_assigned": False,
                            "first_detected": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "seen_in": set([global_track_sources[idx]]) if idx < len(global_track_sources) else set(),
                        }

                    else:
                        global_track_table[global_track_id]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if idx < len(global_track_sources):
                            global_track_table[global_track_id]["seen_in"].add(global_track_sources[idx])

                    snapshot = dict(global_track_table[global_track_id])

                event = {
                    "global_track_id": global_track_id,
                    "event_type": "detected",
                    "first_detected": snapshot["first_detected"],
                    "event_creation_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "is_assigned": snapshot["is_assigned"],
                    "last_update": snapshot["last_update"],
                    "seen_in": list(snapshot["seen_in"]),
                    "description": (
                        f"Tracking event for ID {global_track_id}: "
                        f"assigned={snapshot['is_assigned']}, "
                        f"last_update={snapshot['last_update']}, "
                        f"seen in {list(snapshot['seen_in'])}"
                    ),
                    "deliveries_count": random.randint(0, 100)
                }
                tracking_logs_q.put(event)

            local_global_mapping = {local: global_id for local, global_id in zip(local_track_ids, global_assigned_ids)}
            viz_batch.append((frame, local_global_mapping))

        visualization_queue.put(viz_batch)

    # Cleanup
    with local_state_lock:
        track_rolling_avgs.clear()
    with global_mean_lock:
        global_mean.clear()

    print("[ReID Processor]: Completed batches and flushed remaining mean embeddings.")


