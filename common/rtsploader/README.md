## RTSPChunkLoader
`RTSPChunkLoader` class exposes a document loader for creating video chunks from RTSP streams.
Optionally, you can enable a yolo object detection model to also record detected objects in the chunk documents.

### Yolo disabled
```
from langchain_rtspchunk import RTSPChunkLoader

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>",
    chunk_type="sliding_window", # Traditional sliding window with overlap
    chunk_args={
        "window_size": 85, # Number of frames per chunk
        "fps": 15, # The framerate you save the chunk at
        "overlap": 15, # Number of frames of overlap between consecutive chunks
        "yolo_enabled": False,
    },
    output_dir='output_dir',
)

for doc in rtsp_loader.lazy_load():
    print(f"Sliding Window Chunk metadata: {doc.metadata}")
```

results in
```
{'chunk_path': 'chunk_dir/chunk_2025-06-02_18-29-00.avi, 'timestamp': 2025-06-02_18-29-00, source: 'rtsp://<user>:<pass>@<camera-ip>', 'detected_objects': []}
```


## Yolo enabled
First, download and export ultralyics YOLO object detection model. Alternatively, if you have your own YOLO model, provide the path to the model as argument in RTSPChunkLoader.
```
pip install ultralytics
yolo export model=yolo11n.pt # creates 'yolo11n.pt'
```

```
from langchain_rtspchunk import rtsploader_wrapper

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>",
    chunk_type="sliding_window", # Traditional sliding window with overlap
    chunk_args={
        "window_size": 85, # Number of frames per chunk
        "fps": 15, # The framerate you save the chunk at
        "overlap": 15, # Number of frames of overlap between consecutive chunks
        "yolo_enabled": True,
        "yolo_path": 'yolo11n.pt', # Path to the ultralytics YOLO model
        "yolo_sample_rate": 5 # Every Nth frame is infernced upon
    },
    output_dir='output_dir',
)

for doc in rtsp_loader.lazy_load():
    print(f"Sliding Window Chunk metadata: {doc.metadata}")
```

results in:
```
{'chunk_path': 'output_dir/chunk_2025-06-02_18-29-00.avi, 'timestamp': '2025-06-02_18-29-00', 'source': 'rtsp://<user>:<pass>@<camera-ip>', 'detected_objects': [{'frame 15', objects: ['person', 'chair']}, {'frame 20', objects: ['person', 'chair', 'surfboard']}...]}
```
