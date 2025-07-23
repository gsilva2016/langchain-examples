from typing import Iterator, Dict, List, Tuple
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
import cv2
import time
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import deque
import threading
import queue
import uuid
import torch
import torchvision.transforms as T
from PIL import Image

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class RTSPChunkLoader(BaseLoader):
    def __init__(self, rtsp_url: str, chunk_args: Dict, output_dir: str = "output_chunks"):
        self.rtsp_url = rtsp_url
        self.is_file = os.path.isfile(rtsp_url)
        self.output_dir = output_dir
        self.cap = None
        self.buffer_start_time = None
        self.recording = False  # Flag to track when to save frames
        self.window_size = chunk_args['window_size']
        self.overlap =  chunk_args['overlap']
        self.obj_detect_enabled = chunk_args.get("obj_detect_enabled", False)
        self.frame_id = 0
        self.chunk_queue = queue.Queue() # Use thread safe queue for writing frames
        self.stop_event = threading.Event()
        self.consumer_thread = threading.Thread(target=self._consume_chunks, daemon=True) # Start consumer thread for background buffer writing
        self.consumer_thread.start()
        
        # Object Detection config setup
        if self.obj_detect_enabled:
            # Load OV Model
            from common.rtsploader.dfine_ovinfer import OvInfer
            self.dfine_sample_rate = chunk_args.get("dfine_sample_rate", 5)
            self.dfine_path = chunk_args.get("dfine_path", 'ov_dfine/dfine-s-coco.xml')
            self.model = OvInfer(self.dfine_path)
            print("Loaded DFINE for detection")
            self.detection_threshold = chunk_args.get("detection_threshold", 0.7)
            self.dfine_queue = deque()
            self.detection_results = {}
            self.detection_lock = threading.Lock()
            self.dfine_thread = threading.Thread(target=self._dfine_sliding_window, daemon=True) # Start object detection thread for background inferencing
            self.dfine_thread.start()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _consume_chunks(self):
        while not self.stop_event.is_set():
            try:
                frames, output_path, fps, done_consuming = self.chunk_queue.get(timeout=1)
                if not frames:
                    return
                    
                # Create output file
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Write all frames collected to the output file
                for frame in frames:
                    out.write(frame)
                out.release()
                
                # Indicate consumer saving task is complete
                self.chunk_queue.task_done()

                if done_consuming:
                    done_consuming.set()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Consumer failed: {e}")

    def _sliding_window_chunk(self, frame, current_time, last_chunk=False) -> Tuple[str, str]:
        if not self.frame_buffer:
            self.buffer_start_time = current_time
    
        # Send every Nth frame to YOLO queue
        if self.obj_detect_enabled and self.frame_id % self.dfine_sample_rate == 0 and frame is not None:
            self.dfine_queue.append((self.frame_id, frame))
            
        # Add frame to buffer 
        if frame is not None:
            self.frame_buffer.append((self.frame_id, frame))
            self.frame_id += 1

        # Once frame buffer reached window size, inference frames & consume buffer 
        if len(self.frame_buffer) == self.window_size or last_chunk:
            # Retain start_time, end_time and chunk_id for metadata
            start_time = self.buffer_start_time
            end_time = start_time + (self.window_size / self.fps)
            chunk_id = str(uuid.uuid4())
            
            formatted_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
            chunk_filename = f"chunk_{formatted_time}.mp4"
            chunk_path = os.path.join(self.output_dir, chunk_filename)

            frames_to_save = []
            detected_objects = []

            if self.obj_detect_enabled:
                with self.detection_lock:
                    for i, (fid, f) in enumerate(self.frame_buffer):
                        frames_to_save.append(f)
                        detection = self.detection_results.get(fid)
                        if detection:
                            detected_objects.append({
                                "frame": i,
                                "objects": detection["objects"]
                            })
                            # Clean up memory
                            self.detection_results.pop(fid, None)

            else:
                for i, (fid, f) in enumerate(self.frame_buffer):
                    frames_to_save.append(f)

            # Add consumer synchronization event to prevent path being yielded before video fully written to
            done_consuming = threading.Event()
            # Producer adding frames to Queue, triggering background consumer to write to file
            self.chunk_queue.put((frames_to_save, chunk_path, self.fps, done_consuming))
            # Wait until file is saved before yielding doc
            done_consuming.wait()

            # Remove the oldest frames, excluding overlap
            if not last_chunk:
                frames_to_remove = self.window_size - self.overlap
                for _ in range(frames_to_remove):
                    self.frame_buffer.popleft()

                self.buffer_start_time += frames_to_remove / self.fps
            return chunk_path, formatted_time, start_time, end_time, chunk_id, detected_objects

        return None, None, None, None, None, []
    
    def _dfine_sliding_window(self):
        while not self.stop_event.is_set():
            if not self.dfine_queue:
                time.sleep(0.01)
                continue

            try:
                frame_id, frame = self.dfine_queue.popleft()

                # Process image
                inputs = self.model.process_image(frame, keep_ratio=True)

                # Perform inference on image
                outputs = self.model.infer(inputs)

                # Save bboxes on image for validation
                #self.model.draw_and_save_image(outputs, "rtsp_result.jpg")

                labels = outputs["labels"]
                boxes = outputs["boxes"]
                scores = outputs["scores"]

                objects = []
                for lbl, box, score in zip(labels[0], boxes[0], scores[0]):
                    if score >= self.detection_threshold:
                        class_id = int(lbl)
                        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"id:{class_id}"

                        x1 = box[0] / self.model.ratio
                        y1 = box[1] / self.model.ratio
                        x2 = box[2] / self.model.ratio
                        y2 = box[3] / self.model.ratio

                        objects.append({
                            "label": class_name,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })

                with self.detection_lock:
                    if objects:
                        self.detection_results[frame_id] = {
                            "frame_id": frame_id,
                            "objects": objects
                        }

            except Exception as e:
                print(f"[D-FINE OV ERROR] Inference failed: {e}")


    def _format_times(self, start_time, end_time):
        if self.is_file:
            return f"{start_time:.2f}s", f"{end_time:.2f}s"
        else:
            return datetime.fromtimestamp(start_time).isoformat(), datetime.fromtimestamp(end_time).isoformat()

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load RTSP stream chunks as LangChain Documents."""
        print(f"[INFO] Starting RTSP stream ingestion")

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)  # Use ffmpeg as backend for stable decoding
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open RTSP stream with FFMPEG backend.")
        except Exception as e:
            print("[ERROR] FFMPEG backend is not available, please ensure FFMPEG is installed and accessible.")
            return

        # Get fps from video/stream and convert properties from seconds to frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.window_size = int(self.window_size * self.fps)
        self.overlap = int(self.overlap * self.fps)
        self.frame_buffer = deque(maxlen=self.window_size + self.overlap) # Use a deque for effeciency
        
        # Check if video is file in order to add proper frame delay for object detection
        if self.is_file:
            print(f"[INFO] Processing local video file: {self.rtsp_url}")
            frame_delay = 0.5 / self.fps
        else:
            frame_delay = 0

        try:
            while True:
                start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] Stream ended or error reading frame.")
                    break

                if self.is_file:
                    # Record relative time in seconds for video file
                    current_time = self.frame_id / self.fps
                else:
                    # Record time when the cap starts reading frames from rtsp url
                    current_time = time.time()

                # Use sliding window for ingestion scheme
                chunk_path, formatted_time, start_time, end_time, chunk_id, detections = self._sliding_window_chunk(frame, current_time)
                if chunk_path and formatted_time:
                    start_time, end_time = self._format_times(start_time, end_time)
                    yield Document(
                        page_content=f"Processed RTSP chunk saved at {chunk_path}",
                        metadata={
                            "chunk_id": chunk_id,
                            "chunk_path": chunk_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "source": self.rtsp_url,
                            "detected_objects": detections
                        }
                    )

                elapsed = time.time() - start
                if frame_delay > elapsed:
                    time.sleep(frame_delay - elapsed)

        finally:
            # Process all remaining frames in the buffer
            if len(self.frame_buffer) > 0:

                if self.is_file:
                    # Record relative time in seconds for video file
                    current_time = self.frame_id / self.fps
                else:
                    # Record time when the cap starts reading frames from rtsp url
                    current_time = time.time()

                # Use sliding window for ingestion scheme
                chunk_path, formatted_time, start_time, end_time, chunk_id, detections = self._sliding_window_chunk(None, current_time, last_chunk=True)
                if chunk_path and formatted_time:
                    start_time, end_time = self._format_times(start_time, end_time)
                    yield Document(
                        page_content=f"Processed RTSP chunk saved at {chunk_path}",
                        metadata={
                            "chunk_id": chunk_id,
                            "chunk_path": chunk_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "source": self.rtsp_url,
                            "detected_objects": detections
                        }
                    )

            self.cap.release()
            self.stop_event.set()
            self.consumer_thread.join()
            if self.obj_detect_enabled:
                print(f"Dfine thread joining!")
                self.dfine_thread.join()
            print("[INFO] RTSP stream processing complete.")