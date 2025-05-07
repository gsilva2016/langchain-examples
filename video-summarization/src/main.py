import argparse
import ast
import os
import queue
import sys
import time
from concurrent.futures.thread import ThreadPoolExecutor

from langchain.prompts import PromptTemplate
from langchain_videochunk import VideoChunkLoader

from ov_lvm_wrapper import OVMiniCPMV26Worker
from langchain_summarymerge_score import SummaryMergeScoreTool
from workers import get_sampled_frames, send_summary_request, ingest_summaries_into_milvus, generate_chunk_summaries, ingest_frames_into_milvus
from common.milvus.milvus_wrapper import MilvusManager

def output_handler(text: str,
                   filename: str = '',
                   mode: str = 'w',
                   verbose: bool = True):
    # Print to terminal
    if verbose:
        print(text)

    # Write to file, if requested
    if filename != '':
        with open(filename, mode) as FH:
            print(text, file=FH)


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

    tot_st_time = time.time()
    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()

    # Create template for inputs
    '''prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )

    # Wrap OpenVINO-GenAI optimized model in custom langchain wrapper
    resolution = [] if not args.resolution else args.resolution
    ov_minicpm = OVMiniCPMV26Worker(model_dir=args.model_dir,
                                    device=args.device,
                                    max_new_tokens=args.max_new_tokens,
                                    max_num_frames=args.max_num_frames,
                                    resolution=resolution)

    # Create pipeline and invoke
    chain = prompt | ov_minicpm

    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap)

    # Start log
    output_handler("python " + " ".join(sys.argv),
                   filename=args.outfile, mode='w',
                   verbose=False)

    # Loop through docs and generate chunk summaries    
    chunk_summaries = {}
    for doc in loader.lazy_load():
        # Log metadata
        output_handler(str(f"Chunk Metadata: {doc.metadata}"),
                       filename=args.outfile, mode='a')
        output_handler(str(f"Chunk Content: {doc.page_content}"),
                       filename=args.outfile, mode='a')

        # Generate summaries
        chunk_st_time = time.time()
        video_name = Path(doc.metadata['chunk_path'])
        inputs = {"video": video_name, "question": args.prompt}
        output = chain.invoke(inputs)

        # Log output
        output_handler(output, filename=args.outfile, mode='a', verbose=False)
        chunk_summaries[Path(doc.metadata[
                                 'chunk_path']).stem] = f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output

        output_handler("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time), filename=args.outfile,
                       mode='a')'''

    ingest_queue = queue.Queue()
    summary_queue = queue.Queue()
    frame_queue = queue.Queue()
    
    # Initialize Milvus
    milvus_manager = MilvusManager()
    
    # Summarize the full video, using the subsections summaries from each chunk
        # Two ways to get overall_summary and anomaly score:
        # Method 1. Post an HTTP request to call API wrapper for summary merger (shown below)
    
        # Method 2. Pass existing minicpm based chain, this does not use the FastAPI route and calls the class functions directly
        # summary_merger = SummaryMergeScoreTool(chain=chain, device="GPU")
        # res = summary_merger.invoke({"summaries": chunk_summaries})
    
    overall_summ_st_time = time.time()

    with ThreadPoolExecutor() as pool:
        print("Main: Starting RTSP camera streamer")
        
        print("Main: Getting sampled frames")    
        pool.submit(get_sampled_frames, frame_queue)
        
        print("Main: Starting frame ingestion into Milvus")
        frame_future = pool.submit(ingest_frames_into_milvus, frame_queue, milvus_manager)
        
        print("Main: Starting chunk summary generation")
        cs_future = pool.submit(generate_chunk_summaries, ingest_queue, summary_queue, frame_queue)

        print("Main: Starting chunk summary ingestion into Milvus")
        # Ingest chunk summaries into the running Milvus instance
        milvus_future = pool.submit(ingest_summaries_into_milvus, ingest_queue, milvus_manager)
                        
        print("Main: Starting chunk summary merger")
        # Method 1: Post an HTTP request to call API wrapper for summary merger
        merge_future = pool.submit(send_summary_request, summary_queue)
    
    

    # output_handler(output, filename=args.outfile, mode='a', verbose=False)
