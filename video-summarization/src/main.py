import argparse
import os
import queue
import time
from concurrent.futures.thread import ThreadPoolExecutor
from dotenv import load_dotenv

from workers import get_sampled_frames, send_summary_request, ingest_summaries_into_milvus, generate_chunk_summaries, \
    ingest_frames_into_milvus, generate_chunks, generate_deepsort_tracks, process_reid_embeddings, process_tracking_logs, show_tracking_data, visualize_tracking_data
from common.milvus.milvus_wrapper import MilvusManager
from langchain_openvino_multimodal import OpenVINOBlipEmbeddings


if __name__ == '__main__':
    # Parse inputs
    parser_txt = "Video Pipeline using LangChain, OVMS, DeepSort Tracking/REID, MiniCPM-V-2_6 and LLAMA-3.2-3B-Instruct\n"
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    parser.add_argument("-t", "--max_new_tokens", type=int,
                        help="Maximum number of tokens to be generated.",
                        default=500)
    parser.add_argument("-f", "--max_num_frames", type=int,
                        help="Maximum number of frames to be sampled per chunk for inference. Set to a smaller number if OOM.",
                        default=32)
    parser.add_argument("-c", "--chunk_duration", type=int,
                        help="Maximum length in seconds for each chunk of video.",
                        default=30)
    parser.add_argument("-v", "--chunk_overlap", type=int,
                        help="Overlap in seconds between chunks of input video.",
                        default=2)
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="Desired spatial resolution of input video if different than original. Width x Height")

    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()

    # Load environment variables
    load_dotenv()
    run_vlm = os.getenv("RUN_VLM_PIPELINE", "TRUE").upper() == "TRUE"
    run_reid = os.getenv("RUN_REID_PIPELINE", "TRUE").upper() == "TRUE"
    save_reid_videos = os.getenv("SAVE_REID_VIZ_VIDEOS", "FALSE").upper() == "TRUE"
    overwrite_milvus_collections = os.getenv("OVERWRITE_MILVUS_COLLECTION", "FALSE").upper() == "TRUE"
    print(f"Run VLM Pipeline: {run_vlm}, Run REID Pipeline: {run_reid}, Save REID Videos: {save_reid_videos}, Overwrite Milvus Collections: {overwrite_milvus_collections}")
    
    chunking_mechanism = os.getenv("CHUNKING_MECHANISM", "sliding_window")
    obj_detect_enabled = os.getenv("OBJ_DETECT_ENABLED", "TRUE").upper() == "TRUE"
    obj_detect_model_path = os.getenv("OBJ_DETECT_MODEL_PATH", "ov_dfine/dfine-s-coco.xml")
    obj_detect_sample_rate = int(os.getenv("OBJ_DETECT_SAMPLE_RATE", 5))
    obj_detect_threshold = float(os.getenv("OBJ_DETECT_THRESHOLD", 0.7))
    tracker_det_model_path = os.getenv("TRACKER_DET_MODEL_PATH", "tracker_models/person-detection-0202/FP16/person-detection-0202.xml")
    tracker_reid_model_path = os.getenv("TRACKER_REID_MODEL_PATH", "tracker_models/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml")
    tracker_device = os.getenv("TRACKER_DEVICE", "AUTO")
    tracker_nn_budget = int(os.getenv("TRACKER_NN_BUDGET", 100))
    tracker_max_cosine_distance = float(os.getenv("TRACKER_MAX_COSINE_DISTANCE", 0.5))
    tracker_metric_type = os.getenv("TRACKER_METRIC_TYPE", "cosine")
    tracker_max_iou_distance = float(os.getenv("TRACKER_MAX_IOU_DISTANCE", 0.7))
    tracker_max_age = int(os.getenv("TRACKER_MAX_AGE", 100))
    tracker_n_init = int(os.getenv("TRACKER_N_INIT", 1))
    tracker_width = int(os.getenv("TRACKER_WIDTH", 700))
    tracker_height = int(os.getenv("TRACKER_HEIGHT", 450))
    tracker_dim = (tracker_width, tracker_height)
    tracker_det_thresh = float(os.getenv("TRACKER_DET_THRESH", 0.5))

    # Create queues for inter-thread communication
    chunk_queue = queue.Queue()
    milvus_frames_queue = queue.Queue()
    milvus_summaries_queue = queue.Queue()
    vlm_queue = queue.Queue()
    merger_queue = queue.Queue()
    tracking_chunk_queues = {}
    tracking_results_queues = {}
    visualization_queues = {}
    tracking_logs_queue = queue.Queue()
    
    # Initialize Milvus
    milvus_manager = MilvusManager()
    collection_configs = [
        (os.getenv("VIDEO_COLLECTION_NAME", "video_chunks"), 256),
        (os.getenv("REID_COLLECTION_NAME", "reid_data"), 256),
        (os.getenv("TRACKING_COLLECTION_NAME", "tracking_logs"), 256)
    ]
    for collection_name, dim in collection_configs:
        milvus_manager.create_collection(
            collection_name=collection_name,
            dim=dim,
            overwrite=overwrite_milvus_collections  # overwrite = True if collection already exists and you want to delete, default False
        )
        
    # Initialize Embedding Model
    embedding_model = os.getenv("EMBEDDING_MODEL", "Salesforce/blip-itm-base-coco")
    txt_embedding_device = os.getenv("TXT_EMBEDDING_DEVICE", "GPU")
    img_embedding_device = os.getenv("IMG_EMBEDDING_DEVICE", "GPU")
    ov_blip_embedder = OpenVINOBlipEmbeddings(model_id=embedding_model, ov_text_device=txt_embedding_device,
                                                        ov_vision_device=img_embedding_device)

    # Global track ID table
    global_track_ids = {}
    
    # Add a holder to store chunk tracking events for synchronization between tracking and VLM threads
    chunk_tracking_events = {}
    
    # Video files or RTSP streams
    videos = {
        "video_1": args.video_file,
        # "video_2": 
    }
    
    # Initialize Queues
    for video_id, video in videos.items():
        tracking_results_queues[video_id] = queue.Queue()
        tracking_chunk_queues[video_id] = queue.Queue()
        visualization_queues[video_id] = queue.Queue()
    
    # Create mapping from video_path to tracking_chunk_queue for routing
    video_path_to_tracking_queue = {video: tracking_chunk_queues[video_id] for video_id, video in videos.items()}
    
    futures = []
    with ThreadPoolExecutor() as pool:
        print("[Main]: Starting RTSP camera streamer")
        for video_id, video in videos.items():
            futures.append(pool.submit(
                generate_chunks,
                video,
                args.chunk_duration,
                args.chunk_overlap,
                chunk_queue,
                obj_detect_enabled,
                obj_detect_model_path,
                obj_detect_sample_rate,
                obj_detect_threshold,
                chunking_mechanism,
                chunk_tracking_events if run_reid else None
            ))

        reid_futures = []
        viz_futures = []
        tracking_futures = []
        generate_logs_future = None
        process_logs_future = None
        if run_reid:
            print("[Main]: Running Re-ID and Tracking")
            # Start DeepSORT tracking workers for each video source
            print("[Main]: Starting DeepSORT tracking workers")

            for video_id, video in videos.items():
                tracking_futures.append(pool.submit(
                    generate_deepsort_tracks,
                    tracking_chunk_queues[video_id],
                    tracking_results_queues[video_id],
                    tracker_det_model_path,
                    tracker_reid_model_path,
                    tracker_device,
                    tracker_nn_budget,
                    tracker_max_cosine_distance,
                    tracker_metric_type,
                    tracker_max_iou_distance,
                    tracker_max_age,
                    tracker_n_init,
                    tracker_dim,
                    det_thresh=tracker_det_thresh,
                    write_video=False,
                    chunk_tracking_events=chunk_tracking_events
                ))
            
            # Process re-id embeddings
            print("[Main]: Starting re-id embedding processing")

            for video_id, video in videos.items():
                reid_futures.append(pool.submit(process_reid_embeddings, tracking_results_queues[video_id], tracking_logs_queue, visualization_queues[video_id], global_track_ids, milvus_manager, "reid_data"))
                if save_reid_videos:
                    viz_futures.append(pool.submit(visualize_tracking_data, visualization_queues[video_id], tracker_dim))

            # Show tracking data
            print("[Main]: Show tracking data")
            generate_logs_future = pool.submit(show_tracking_data, global_track_ids)

            # Process tracking logs
            print("[Main]: Starting tracking logs processing")
            process_logs_future = pool.submit(process_tracking_logs, tracking_logs_queue, milvus_manager, ov_blip_embedder)
        
        if run_vlm:
            print("[Main]: Getting sampled frames")    
            sample_future = pool.submit(get_sampled_frames, chunk_queue, milvus_frames_queue, vlm_queue, 
                                    video_path_to_tracking_queue, args.max_num_frames, save_frame=False,
                                    resolution=args.resolution)
        
            print("[Main]: Starting frame ingestion into Milvus")
            milvus_frames_future = pool.submit(ingest_frames_into_milvus, milvus_frames_queue, milvus_manager, ov_blip_embedder)
                        
            print("[Main]: Starting chunk summary generation")
            cs_future = pool.submit(
                generate_chunk_summaries,
                vlm_queue,
                milvus_summaries_queue,
                merger_queue,
                args.prompt,
                args.max_new_tokens,
                obj_detect_enabled,
                milvus_manager,
                run_reid,  # tracking_enabled
                chunk_tracking_events if run_reid else None
            )

            print("[Main]: Starting chunk summary ingestion into Milvus")
            milvus_summaries_future = pool.submit(ingest_summaries_into_milvus, milvus_summaries_queue, milvus_manager, ov_blip_embedder)       
            
            # Summarize the full video, using the subsections summaries from each chunk
            # Post an HTTP request to OVMS for summary merger (shown below)
            print("[Main]: Starting chunk summary merger")
            merge_future = pool.submit(send_summary_request, merger_queue)

        for future in futures:
            future.result()

        chunk_queue.put(None)
        
        for video_id, video in videos.items():
            tracking_chunk_queues[video_id].put(None)
                
        sample_future.result()
        
        if run_vlm:
            milvus_frames_future.result()
            cs_future.result()
            milvus_summaries_future.result()
            merge_future.result()

        for tf in tracking_futures:
            tf.result()
            
        for reid_future in reid_futures:
            reid_future.result()
            
        for viz_future in viz_futures:
            viz_future.result()

        # Set sentinel BEFORE waiting for show_tracking_data to exit
        if generate_logs_future:
            global_track_ids[-1] = {}
            generate_logs_future.result()
        
        tracking_logs_queue.put(None)

        if process_logs_future:
            process_logs_future.result()

        print("[Main]: All tasks completed")
