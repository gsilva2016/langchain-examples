## RTSPChunkLoader
`RTSPChunkLoader` class exposes a document loader for creating video chunks from RTSP streams.
Optionally, you can enable a D-FINE object detection model to record detected objects in the frames to the chunk documents.

### Object Detection disabled
```
from rtsploader_wrapper import RTSPChunkLoader

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>",
    chunk_type="sliding_window", # Traditional sliding window with overlap
    chunk_args={
        "window_size": 85, # Number of frames per chunk
        "fps": 15, # The framerate you save the chunk at
        "overlap": 15, # Number of frames of overlap between consecutive chunks
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
bash download_model.sh  # Creates 'ov_dfine/dfine-l-coco.xml & .bin'
```

```
from rtsploader_wrapper import RTSPChunkLoader

rtsp_loader = RTSPChunkLoader(
    rtsp_url="rtsp://<user>:<pass>@<camera-ip>",
    chunk_type="sliding_window", # Traditional sliding window with overlap
    chunk_args={
        "window_size": 85, # Number of frames per chunk
        "fps": 15, # The framerate you save the chunk at
        "overlap": 15, # Number of frames of overlap between consecutive chunks
        "obj_detect_enabled": True,
        "dfine_path": 'ov_dfine/dfine-l-coco.xml', # Path to the D-FINE model
        "dfine_sample_rate": 5 # Every Nth frame is infernced upon
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
'detected_objects': [{'frame': 0,
			'objects': [
			    {'label': 'person', 'bbox': [2219.25439453125, 647.7630615234375, 2593.083740234375, 1600.69482421875]},
			    {'label': 'surfboard', 'bbox': [2779.221435546875, 416.277099609375, 3021.228271484375, 1071.7967529296875]},
			    {'label': 'chair', 'bbox': [115.33447265625, 1249.074951171875, 444.9425354003906, 1729.0045166015625]},
			    {'label': 'couch', 'bbox': [106.50575256347656, 1886.9461669921875, 1974.1636962890625, 2147.5087890625]}]}, ...]
```
