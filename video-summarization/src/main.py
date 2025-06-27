import argparse
import os
import queue
from concurrent.futures.thread import ThreadPoolExecutor
import time

from workers import get_sampled_frames, send_summary_request, ingest_summaries_into_milvus, generate_chunk_summaries, ingest_frames_into_milvus, generate_chunks
from common.milvus.milvus_wrapper import MilvusManager


if __name__ == '__main__':
    # Parse inputs
    parser_txt = "Video Summarization using LangChain, OVMS, MiniCPM-V-2_6 and LLAMA-3.2-3B-Instruct\n"
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
    parser.add_argument("--camera_fps", type=int,
                        help="Frames per second of the camera/video.",
                        default=15)
    parser.add_argument("--yolo_enabled", action="store_true",
                        help="Enable YOLO-based chunking.")
    parser.add_argument("--yolo_path", type=str,
                        help="Path to YOLO model weights.",
                        default="yolo11n.pt")
    parser.add_argument("--yolo_sample_rate", type=int,
                        help="Sample rate for YOLO detection.",
                        default=5)
    parser.add_argument("--chunking_mechanism", type=str,
                        help="Chunking mechanism to use (e.g., sliding_window, fixed).",
                        default="sliding_window")

    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()

    chunk_queue = queue.Queue()
    milvus_frames_queue = queue.Queue()
    milvus_summaries_queue = queue.Queue()
    vlm_queue = queue.Queue()
    merger_queue = queue.Queue()
    
    # Initialize Milvus
    milvus_manager = MilvusManager()
    
    # Video files or RTSP streams
    videos = {
        "video_1": args.video_file, 
        # "video_2": second file
    }
    
    futures = []    
    with ThreadPoolExecutor() as pool:
        print("Main: Starting RTSP camera streamer")
        for video in videos.values():
            futures.append(pool.submit(
                generate_chunks,
                video,
                args.chunk_duration,
                args.chunk_overlap,
                chunk_queue,
                args.camera_fps,
                args.yolo_enabled,
                args.yolo_path,
                args.yolo_sample_rate,
                args.chunking_mechanism
            ))
        
        print("Main: Getting sampled frames")    
        sample_future = pool.submit(get_sampled_frames, chunk_queue, milvus_frames_queue, vlm_queue, args.max_num_frames, save_frame=False,
                                    resolution=args.resolution)
        
        print("Main: Starting frame ingestion into Milvus")
        milvus_future = pool.submit(ingest_frames_into_milvus, milvus_frames_queue, milvus_manager)
        
        print("Main: Starting chunk summary generation")
        cs_future = pool.submit(generate_chunk_summaries, vlm_queue, milvus_summaries_queue, merger_queue, args.prompt, args.max_new_tokens, args.yolo_enabled)

        print("Main: Starting chunk summary ingestion into Milvus")
        milvus_future = pool.submit(ingest_summaries_into_milvus, milvus_summaries_queue, milvus_manager)                
        
        # Summarize the full video, using the subsections summaries from each chunk
        # Post an HTTP request to OVMS for summary merger (shown below)
        print("Main: Starting chunk summary merger")
        merge_future = pool.submit(send_summary_request, merger_queue)
    
        while not all([future.done() for future in futures]):
            time.sleep(0.1)
        
        chunk_queue.put(None)
        
        merge_future.result()
        print("Main: All tasks completed")
        print("Main: Lastly, waiting for merge task to complete")
