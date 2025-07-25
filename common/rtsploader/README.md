## RTSPChunkLoader
`RTSPChunkLoader` class exposes a document loader for creating video chunks from RTSP streams.
Optionally, you can enable a D-FINE object detection model to record detected objects in the frames to the chunk documents.

### Object Detection disabled
```
from rtsploader_wrapper import RTSPChunkLoader

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>", # Can also be a local video file path
    chunk_args={
        "window_size": 30, # Number of seconds per chunk
        "overlap": 2, # Number of seconds of overlap between consecutive chunks
        "obj_detect_enabled": False,
    },
    output_dir='cam_1',
)

for doc in rtsp_loader.lazy_load():
    print(f"Sliding Window Chunk metadata: {doc.metadata}")
```

results in:
```
Sliding Window Chunk metadata:
{'chunk_id': '5b58wefoih234h334j',
'chunk_path': 'cam_1/chunk_2025-06-02_18-29-00.avi',
'start_time': '2025-06-02T18:29:00.315088',
'end_time': '2025-06-02T18:29:05.412887',
source: 'rtsp://<user>:<pass>@<camera-ip>',
'detected_objects': []}
```


## Object Detection enabled
First, download and export [D-FINE](https://github.com/Peterande/D-FINE/tree/master) object detection model. Alternatively, if you have your own D-FINE model (in OpenVINO format), provide the path to the model as argument in RTSPChunkLoader.
```
bash download_model.sh  # Creates 'ov_dfine/dfine-s-coco.xml & .bin'
```

```
from rtsploader_wrapper import RTSPChunkLoader

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>", # Can also be a local video file path
    chunk_args={
        "window_size": 30, # Number of seconds per chunk
        "overlap": 2, # Number of seconds of overlap between consecutive chunks
        "obj_detect_enabled": True,
        "dfine_path": 'ov_dfine/dfine-s-coco.xml', # Path to the D-FINE model
        "dfine_sample_rate": 5, # Every 5th frame is infernced upon
        "detection_threshold": 0.7 # Only capture objects with detection confidence greater than 0.7

    },
    output_dir='cam_1',
)

for doc in rtsp_loader.lazy_load():
    print(f"Sliding Window Chunk metadata: {doc.metadata}")
```

results in:
```
Sliding Window Chunk metadata: 
{'chunk_id': '3d3h45tfwef34fnb7ug',
'chunk_path': 'cam_1/chunk_2025-06-02_18-29-00.avi',
'start_time': '2025-06-02T18:29:00.315088',
'end_time': '2025-06-02T18:29:05.412887',
'source': 'rtsp://<user>:<pass>@<camera-ip>',
'detected_objects': [{'frame': 0, 'objects': [
			{'label': 'person', 'bbox': [2219.25, 647.76, 2593.08, 1600.69]},
			{'label': 'surfboard', 'bbox': [2779.22, 416.27, 3021.22, 1071.79]},
			{'label': 'chair', 'bbox': [115.33, 1249.07, 444.94, 1729.00]},
			{'label': 'couch', 'bbox': [106.50, 1886.94, 1974.16, 2147.50]}]}, ...]
```
