from datetime import datetime
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
from dotenv import load_dotenv
from langchain_summarymerge_score import SummaryMergeScoreTool
from common.rtsploader.rtsploader_wrapper import RTSPChunkLoader
from common.sampler.frame_sampler import FrameSampler

import requests
from PIL import Image
import io
import base64
import time

load_dotenv()
OVMS_ENDPOINT = os.environ.get("OVMS_ENDPOINT", None)
VLM_MODEL = os.environ.get("VLM_MODEL", "openbmb/MiniCPM-V-2_6")

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
            print(f"Summary Merger: Received {len(summaries)} chunk summaries for merging")
            formatted_req = {
                "summaries": {chunk["chunk_id"]: chunk["chunk_summary"] for chunk in summaries}
            }
            print(f"Summary Merger: Sending {len(summaries)} chunk summaries for merging")
            try:
                merge_res = summary_merger.invoke(formatted_req)
                print(f"Overall Summary: {merge_res['overall_summary']}")
                print(f"Anomaly Score: {merge_res['anomaly_score']}")
            except Exception as e:
                print(f"Summary Merger: Request failed: {e}")

            summaries = []

        if last and not summaries:
            print("Summary Merger: All summaries processed, exiting.")
            return
    
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

def ingest_summaries_into_milvus(milvus_summaries_q: queue.Queue, milvus_manager: object):
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
            print(f"Milvus: Ingesting {len(summaries)} chunk summaries into Milvus")
            try:
                response = milvus_manager.embed_txt_and_store(summaries)
                print(f"Milvus: Chunk Summaries Ingested into Milvus: {response['status']}, Total chunks: {response['total_chunks']}")
            except Exception as e:
                print(f"Milvus: Chunk Summaries Ingestion Request failed: {e}")
            summaries = []  

        if last and not summaries:
            print("Milvus: All summaries ingested, exiting.")
            break

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
            "detected_objects": chunk["detected_objects"]
        }
        
        milvus_summaries_queue.put(chunk_summary)
        merger_queue.put(chunk_summary)

    print("VLM: Ending service")
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
        print(f"CHUNK LOADER: Chunking video: {video_path} and chunk path: {doc.metadata['chunk_path']}")
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
            print("CHUNK LOADER: placing in tracking queue")            
            tracking_chunk_queue.put(chunk)
    print(f"CHUNK LOADER: Chunk generation completed for {video_path}")
    
def call_vertex():
    pass

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
            if sampling_rate <= 1:
                max_num_frames = int(len(vr))  # sample every frame
            else:
                max_num_frames = max(1, int(len(vr) / sampling_rate))  # sample every Nth frame
            sampler = FrameSampler(max_num_frames=max_num_frames, resolution=list(resize_dim), save_frame=False)
        print(f"DeepSORT: Processing {max_num_frames} frames from {video_path}") 

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
        for i, frame in enumerate(frames):
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

            # Print aligned features for each output track
            # for i, out_track in enumerate(outputs):
            #     print(f"Track {out_track[-1]} features: {len(reid_features[i])}")

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

            # Accumulate tracking results for this frame
            chunk_tracking_results.append({
                "chunk_id": chunk["chunk_id"],
                "bbox_xyxy": bbox_xyxy,
                "identities": identities,
                "reid_features": reid_features,
                "video_path": chunk["video_path"],
                "chunk_path": chunk["chunk_path"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"]
            })            
            
        # Release video writer if enabled
        if out is not None:
            out.release()

        # Put the list of tracking results for the chunk in the results queue
        tracking_results_queue.put(chunk_tracking_results)
        end_t = time.time() - start_t
        print(f"DeepSORT: Finshed processing {video_path}: {len(frames)} frames in {end_t} sec")

    # Signal completion by putting None in the results queue        
    print("DeepSORT: Tracking completed")
    tracking_results_queue.put(None)
