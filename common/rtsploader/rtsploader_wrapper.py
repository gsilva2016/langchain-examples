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

class RTSPChunkLoader(BaseLoader):
    def __init__(self, rtsp_url: str, chunk_type: str, chunk_args: Dict, output_dir: str = "output_chunks"):
        self.rtsp_url = rtsp_url
        self.chunk_type = chunk_type
        self.output_dir = output_dir
        self.cap = None
        self.buffer_start_time = None
        self.recording = False  # Flag to track when to save frames
        self.fps = chunk_args.get("fps", 15)  # Safe and defaulted (shared configuration)
        self.window_size = chunk_args['window_size']
        self.overlap =  chunk_args['overlap']
        self.yolo_enabled = chunk_args.get("yolo_enabled", False)
        self.frame_id = 0
        self.frame_buffer = deque(maxlen=self.window_size + self.overlap) # Use a deque for effeciency
        self.chunk_queue = queue.Queue() # Use thread safe queue for writing frames
        self.stop_event = threading.Event()
        self.consumer_thread = threading.Thread(target=self._consume_chunks, daemon=True) # Start consumer thread for background buffer writing
        self.consumer_thread.start()
        
        # yolo config setup
        if self.yolo_enabled:
            from ultralytics import YOLO
            self.yolo_sample_rate = chunk_args.get("yolo_sample_rate", 5)  # Run inference every Nth frame
            self.yolo_path = chunk_args.get("yolo_path", "yolo11n.pt") # Default to using OV yolov11n
            self.model = YOLO(self.yolo_path)
            self.yolo_queue = deque()
            self.detection_results = {}
            self.detection_lock = threading.Lock()
            self.yolo_thread = threading.Thread(target=self._yolo_sliding_window, daemon=True) # Start yolo thread for background inferencing
            self.yolo_thread.start()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _consume_chunks(self):
        while not self.stop_event.is_set():
            try:
                frames, output_path, fps = self.chunk_queue.get(timeout=1)
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
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Consumer failed: {e}")

    def _sliding_window_chunk(self, frame, current_time) -> Tuple[str, str]:
        if not self.frame_buffer:
            self.buffer_start_time = current_time
    
        # Send every Nth frame to YOLO queue
        if self.yolo_enabled and self.frame_id % self.yolo_sample_rate == 0:
            self.yolo_queue.append((self.frame_id, frame))
            
        # Add frame to buffer 
        self.frame_buffer.append((self.frame_id, frame))
        self.frame_id += 1

        # Once frame buffer reached window size, inference frames & consume buffer 
        if len(self.frame_buffer) == self.window_size:
            # Retain start_time, end_time and chunk_id for metadata
            start_time = self.buffer_start_time
            end_time = start_time + (self.window_size / self.fps)
            chunk_id = str(uuid.uuid4())
            
            formatted_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
            chunk_filename = f"chunk_{formatted_time}.mp4"
            chunk_path = os.path.join(self.output_dir, chunk_filename)

            frames_to_save = []
            detected_objects = []

            if self.yolo_enabled:
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

            # Producer adding frames to Queue, triggering background consumer to write to file
            self.chunk_queue.put((frames_to_save, chunk_path, self.fps))

            # Remove the oldest frames, excluding overlap
            frames_to_remove = self.window_size - self.overlap
            for _ in range(frames_to_remove):
                self.frame_buffer.popleft()

            self.buffer_start_time += frames_to_remove / self.fps
            return chunk_path, formatted_time, start_time, end_time, chunk_id, detected_objects

        return None, None, None, None, None, []
    
    def _yolo_sliding_window(self):
        while not self.stop_event.is_set():
            if not self.yolo_queue:
                time.sleep(0.01)
                continue

            try:
                # Get first frame (FIFO)
                frame_id, frame = self.yolo_queue.popleft()

                # Perform inference using YOLO
                results = self.model(frame, verbose=False)
                detected_classes = []

                for result in results:
                    for box in result.boxes.data:
                        class_id = int(box[5].item())
                        class_name = self.model.names[class_id]
                        detected_classes.append(class_name)

                with self.detection_lock:
                    self.detection_results[frame_id] = {
                        "frame_id": frame_id,
                        "objects": detected_classes
                    }

            except Exception as e:
                print(f"[YOLO ERROR] Inference failed: {e}")

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
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] Stream ended or error reading frame.")
                    break

                # Record time when the cap starts reading frames from rtsp url
                current_time = time.time()

                # Use sliding window for ingestion scheme
                chunk_path, formatted_time, start_time, end_time, chunk_id, detections = self._sliding_window_chunk(frame, current_time)

                if chunk_path and formatted_time:
                    yield Document(
                        page_content=f"Processed RTSP chunk saved at {chunk_path}",
                        metadata={
                            "chunk_id": chunk_id,
                            "chunk_path": chunk_path,
                            "start_time": datetime.fromtimestamp(start_time).isoformat(),
                            "end_time": datetime.fromtimestamp(end_time).isoformat(),
                            "source": self.rtsp_url,
                            "detected_objects": detections
                        },
                    )
        finally:
            self.cap.release()
            self.stop_event.set()
            self.consumer_thread.join()
            if self.yolo_enabled:
                self.yolo_thread.join()
            print("[INFO] RTSP stream processing complete.")
