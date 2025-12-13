from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import uuid
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from common.milvus.milvus_wrapper import MilvusManager

env_path = os.environ.get("ENV_PATH", None)
if env_path is None:
    print("No ENV_PATH set, values might be missing.")
else:
    print(f"Loading environment from: {env_path}")
    load_dotenv(env_path)

# Thresholds
DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", 0.8))
SIM_SCORE_THRESHOLD = float(os.environ.get("REID_SIM_SCORE_THRESHOLD", 0.65))
TOO_SIMILAR_THRESHOLD = float(os.environ.get("TOO_SIMILAR_THRESHOLD", 0.95))
AMBIGUITY_MARGIN = float(os.environ.get("AMBIGUITY_MARGIN", 0.15))
PARTITION_CREATION_INTERVAL = int(os.environ.get("PARTITION_CREATION_INTERVAL", 1))

# Global locks
global_track_locks = defaultdict(threading.Lock)
global_assignment_lock = threading.Lock()

# Shared across threads
track_rolling_avgs = defaultdict(lambda: deque(maxlen=8))
global_mean = {}

# Locks for concurrency
local_state_lock = threading.Lock()
global_mean_lock = threading.Lock()

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
                    "db_entry_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

def flush_final_embeddings(milvus_manager: MilvusManager, collection_name: str = "reid_data"):
    """
    Flush any remaining embeddings in rolling averages to Milvus.
    """
    with global_assignment_lock:
        flush_embeddings, flush_metadatas = [], []
        
        
        for gid, mean in list(global_mean.items()):
            flush_embeddings.append(mean.tolist())
            flush_metadatas.append({
                "global_track_id": gid,
                "mode": "reid",
                "db_entry_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "final_flush"
            })
            
        if flush_embeddings:
            milvus_manager.insert_data(
                collection_name=collection_name,
                vectors=flush_embeddings,
                metadatas=flush_metadatas
            )

def clear_reid_state():
    """
        Clear the in-memory state for ReID tracking.
    """
    with local_state_lock:
        track_rolling_avgs.clear()
    
    with global_mean_lock:
        global_mean.clear()