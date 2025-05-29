import argparse
import os
import queue
from concurrent.futures.thread import ThreadPoolExecutor
import time

from langchain_summarymerge_score import SummaryMergeScoreTool
from workers import get_sampled_frames, send_summary_request, ingest_summaries_into_milvus, generate_chunk_summaries, ingest_frames_into_milvus, generate_chunks
from common.milvus.milvus_wrapper import MilvusManager


if __name__ == '__main__':
    # Parse inputs
    parser_txt = "Generate video summarization using LangChain, OpenVINO-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("model_dir", type=str,
                        help="Path to openvino-genai optimized model")
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    parser.add_argument("-d", "--device", type=str,
                        help="Target device for running ov MiniCPM-v-2_6",
                        default="CPU")
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
    parser.add_argument("-o", "--outfile", type=str,
                        help="File to write generated text.", default='')

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
            futures.append(pool.submit(generate_chunks, video, args.chunk_duration, args.chunk_overlap, 
                        chunk_queue, chunking_mechanism="sliding_window"))
        
        print("Main: Getting sampled frames")    
        sample_future = pool.submit(get_sampled_frames, chunk_queue, milvus_frames_queue, vlm_queue, args.max_num_frames, save_frame=False)
        # futures.append(sample_future)
        
        print("Main: Starting frame ingestion into Milvus")
        milvus_future = pool.submit(ingest_frames_into_milvus, milvus_frames_queue, milvus_manager)
        # futures.append(milvus_future)
        
        print("Main: Starting chunk summary generation")
        cs_future = pool.submit(generate_chunk_summaries, vlm_queue, milvus_summaries_queue, merger_queue)
        # futures.append(cs_future)

        print("Main: Starting chunk summary ingestion into Milvus")
        # Ingest chunk summaries into the running Milvus instance
        milvus_future = pool.submit(ingest_summaries_into_milvus, milvus_summaries_queue, milvus_manager)                
        # futures.append(milvus_future)
        
        # Summarize the full video, using the subsections summaries from each chunk
        # Two ways to get overall_summary and anomaly score:
        # Method 1. Post an HTTP request to call API wrapper for summary merger (shown below)
    
        # Method 2. Pass existing minicpm based chain, this does not use the FastAPI route and calls the class functions directly
        # summary_merger = SummaryMergeScoreTool(chain=chain, device="GPU")
        # res = summary_merger.invoke({"summaries": chunk_summaries})
        print("Main: Starting chunk summary merger")
        merge_future = pool.submit(send_summary_request, merger_queue)
        # futures.append(merge_future)
    
        while not all([future.done() for future in futures]):
            time.sleep(0.1)
        
        chunk_queue.put(None)