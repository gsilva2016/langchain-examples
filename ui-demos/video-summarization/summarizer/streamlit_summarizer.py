import argparse
import ast
import os
import sys
import time
import json
from itertools import tee
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import queue
import re
import uuid
import subprocess

import requests
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.video import VideoChunkLoader

from ov_lvm_wrapper import OVMiniCPMV26Worker, reset_chunk_variables
from vertex_extension import VertexWrapper

os.environ["no_proxy"] = "localhost,127.0.0.1"

# Create thread safe queue for merge summaries
merge_queue = queue.Queue()

# Create thread safe queue for vertex summaries
vertex_queue = queue.Queue()

ingest_demo_chunks = False

# Create variable for loading model onto GPU
_model_cache = None

alert_queue = queue.Queue()
stop_signal = threading.Event()

def concatenate_videos(video_path, output_path='merged_video.mp4', list_file='merge_videos.txt'):
    with open(list_file, "w") as f:
        for path in video_path:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print("Error during video concatenation")
        return None
    
def delete_file_if_exists(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")
        else:
            print(f"{path} not found")
    except Exception as e:
        print(f"Error in removing file {path}")

def post_request(input_data):
    formatted_req = {"summaries": input_data}
    response = requests.post(url="http://127.0.0.1:8000/merge_summaries", json=formatted_req)
    return response.content

def ingest_into_milvus(ingest_q):
    while not stop_signal.is_set():
        chunk_summaries = []
        end_ingestion = False
        while not ingest_q.empty():
            item = ingest_q.get()
            if item == "END":
                end_ingestion = True
                ingest_q.task_done()
                break
            chunk_summaries.append(item)

        if end_ingestion:
            break

        if chunk_summaries:
            formatted_req = {
                "data": chunk_summaries
            }

            # print(formatted_req)
            print(f"Milvus: Ingesting {len(chunk_summaries)} chunk summaries into Milvus")
            try:
                response = requests.post(url="http://127.0.0.1:8000/embed_txt_and_store", json=formatted_req)
                if response.status_code != 200:
                    print(f"Milvus: Error in ingest_into_milvus: {response.status_code}, {response.content}")

                milvus_res = response.json()
                print(
                    f"Milvus: Chunk Summaries Ingested into Milvus: {milvus_res['status']}, Total chunks: {milvus_res['total_chunks']}")

            except requests.exceptions.RequestException as e:
                print(f"Milvus: Request failed: {e}")

        else:
            #continue
            print("Milvus: Waiting for chunk summaries to ingest")

        time.sleep(10)

def tag_last(generator):
    gen1, gen2 = tee(generator)
    next(gen2, None)
    for current, next_item in zip(gen1, gen2):
        yield current, False
    yield next_item, True

merge_lock = threading.Lock()  # Add a threading lock

def async_merge_chunks(chunk_summaries, merge_start_time, end_time, outfile, extend_to_vertex, 
                       cloud_model, cloud_prompt, anomaly_thresh, loader, doc, mode="w",
                       processing_chunk_ids=None, merge_threads=None):
    # Check if stop_signal is set before starting   
    if stop_signal.is_set():       
        print("\n\nMerge operation aborted due to stop signal.\n\n")
        if merge_threads:
            for t in merge_threads:
                print(f"Joinging threads in merge stop: {t}")
                t.join()
        return None
    
    try:
        with merge_lock:  # Ensure only one thread accesses the merger at a time
            print('\n\nSending Chunks to Merger!\n\n')
            merge_st_time = time.time()
            with ThreadPoolExecutor() as pool:
                if stop_signal.is_set():       
                    print("\n\nMerge operation aborted due to stop signal.\n\n")
                    if merge_threads:
                        for t in merge_threads:
                            print(f"Joinging threads in merge stop: {t}")
                            t.join()
                    return None
                future = pool.submit(post_request, chunk_summaries)
                merge_res = ast.literal_eval(future.result().decode("utf-8"))
                merge_output = f"[MERGED SUMMARY {merge_start_time}-{end_time}sec]\n{merge_res['overall_summary']}\n\nAnomaly score from LLM: {merge_res['anomaly_score']}\n\n"
                anomaly_match = re.search(r"Anomaly score from LLM:\s*([0-9.]+)", merge_output)
                color = ""
                if anomaly_match:
                    score = float(anomaly_match.group(1))
                    if score < 0.3:
                        color = "green"
                    elif score < 0.7:
                        color = "orange"
                    else:
                        color = "red"
                    styled_scoreline = f'<span style="color:{color}">Anomaly score from LLM: {score}</span>'
                    merge_output = re.sub("Anomaly score from LLM:\s*([0-9.]+)", styled_scoreline, merge_output)

                alert_queue.put(score)
                merge_queue.put(merge_output)
                print(f"Updated merge queue with merged summary: {merge_output}\n")
                print(f"Merge Result from local LLM: {merge_res['overall_summary']}\n")
                print(f"Anomaly score from LLM: {merge_res['anomaly_score']}\n")
            print("Merge Chunks Time: {} sec\n".format(time.time() - merge_st_time))

            # Extend to cloud, if asked
            if extend_to_vertex and merge_res['anomaly_score'] >= anomaly_thresh:
                # Check if stop_signal is set before starting   
                if stop_signal.is_set():       
                    print("\n\nCloud operation aborted due to stop signal.\n\n")
                    if merge_threads:
                        for t in merge_threads:
                            print(f"Joinging threads in merge stop: {t}")
                            t.join()
                    return None

                print("Sending anomalous clip to Vertex!")
                cloud_st_time = time.time()
                
                # Use processing_chunk_ids to calculate chunks to upload
                merged_chunks = [os.path.join(loader.output_dir,
                                              f"chunk_{ch_id}.mp4") for ch_id in processing_chunk_ids]
                #print(f"Created merged chunks: {merged_chunks}")

                merged_path = concatenate_videos(merged_chunks, output_path='merged_video.mp4')
                
                cloud_response = cloud_model.generate(cloud_prompt, video_paths=[merged_path])

                delete_file_if_exists(merged_path)

                print(f"\n\Generation from cloud model: {cloud_response}\n\n")
                anomaly_score = cloud_model.extract_anomaly_score(cloud_response)

                # Update the merge res summary and anomaly score with vertex output
                merge_res = {'overall_summary': cloud_response,
                             'anomaly_score': anomaly_score}
                color = ""
                if anomaly_score < 0.3:
                    color = "green"
                elif anomaly_score < 0.7:
                    color = "orange"
                else:
                    color = "red"
                styled_scoreline = f'<span style="color:{color}">Anomaly score from gemini: {anomaly_score}</span>'
                cloud_response = re.sub(r"\*\*anomaly score\*\*: [-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", styled_scoreline, cloud_response)
                print(f"Cloud response: {cloud_response}")
                cloud_response = f"[CLOUD SUMMARY {merge_start_time}-{end_time}sec]\n{cloud_response}\n\n"
                vertex_queue.put(cloud_response)
                cloud_model.cleanup()
                print("Cloud Summary Time: {} sec\n\n".format(time.time() - cloud_st_time))

            if outfile:
                merge_res["start_time"] = merge_start_time
                merge_res["end_time"] = end_time
                with open(outfile, mode) as FH:
                    json.dump(merge_res, FH, indent=4)
                    FH.write("\n")

            return merge_res
    except Exception as e:
        print(f"Error during merge: {e}")
        return None

def summarizer_main(args):
    reset_chunk_variables()
    init_st_time = time.time()

    # Check if model is already loaded
    global _model_cache
    if _model_cache is None:
        print("Loading model onto GPU...")
        resolution = [] if not args.resolution else args.resolution
        _model_cache = OVMiniCPMV26Worker(
        model_dir=args.model_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_num_frames=args.max_num_frames,
        resolution=resolution
    )
    else:
        print("Using cached model")

    ov_minicpm = _model_cache

    # Check video exists
    if not os.path.exists(args.video_file):
        raise FileNotFoundError(f"{args.video_file} does not exist.")
    
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )

    chain = prompt | ov_minicpm

   # Initialize cloud model
    if args.extend_to_vertex:
        print('Initializing cloud model instance...')
        cloud_model = VertexWrapper(args.cloud_model)
        cloud_prompt = """You are an expert investigator that analyzes video feeds. 
        I want you to call out moments of identified or highly suspected shoplifting. 
        Look at people interacting with objects on display and taking into their possession. Please then provide a score between 0 and 1 
        to represent the amount of suspicious activity you've just analyzed. If you see no humans, then score must be 0, as there is no 
        suspicious activity possible. The score should be a float rounded to the tenth decimal.
        Please organize your answer according to this example:
        Overall Summary: A summary of the entire text description in about five sentences or less, focused on the people rather than the scene itself.
        Potential Suspicious Activity: List any activities that might indicate suspicious behavior.
        **anomaly score**: <floating point value representing suspicious activity>"""
    else:
        print("Not initialzing cloud model instance...")
        cloud_model = None
        cloud_prompt = None

    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap
    )

    merge_cadence = max(1, int(args.merge_cadence / args.chunk_duration)) if args.merge_cadence else float('inf')

    print("\nInitialization Time: {} sec\n".format(time.time() - init_st_time))

    # Loop through docs and generate individual chunk summaries
    mode = "w"
    chunk_summaries = {}
    merge_start_time = 0
    merge_threads = []
    all_chunk_outputs = {}
    ingest_queue = queue.Queue()

    print("Main: Starting chunk summary ingestion into Milvus")

    # Ingest chunk summaries into the running Milvus instance
    global ingest_demo_chunks

    # Ingest chunk summaries into the running Milvus instance
    with ThreadPoolExecutor() as pool:
        if not ingest_demo_chunks:
            milvus_future = pool.submit(ingest_into_milvus, ingest_queue)

        tot_inf_st_time = time.time()

        for doc, is_last in tag_last(loader.lazy_load()):
            if stop_signal.is_set():
                print("Summarizer main - stop signal recieved")
                for t in merge_threads:
                    print(f"Joinging thread: {t}")
                    t.join()
                return

            chunk_st_time = time.time()
            video_name = Path(doc.metadata['chunk_path'])
            inputs = {"video": video_name, "question": args.prompt}
            output = chain.invoke(inputs)

            chunk_key = Path(doc.metadata['chunk_path']).stem
            chunk_summary = {
                "chunk_id": doc.metadata['chunk_id'],
                "summary": f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n{output}"
            }
            chunk_summaries[chunk_key] = chunk_summary
            all_chunk_outputs[chunk_key] = chunk_summary["summary"]

            print(f"Chunk Summary Time: {time.time() - chunk_st_time} sec\n")

            if not ingest_demo_chunks:
                ingest_queue.put(
                    {
                        "chunk_id": f"{doc.metadata['chunk_id']}_{uuid.uuid4()}",
                        "chunk_path": doc.metadata['chunk_path'],
                        "chunk_summary": f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n{output}",
                        "start_time": f"{doc.metadata['start_time']}",
                        "end_time": f"{doc.metadata['end_time']}"

                    }
                )
                print(f"Milvus ingested chunk: {doc.metadata['chunk_id']}")

            call_merger = (doc.metadata['chunk_id']+1) % merge_cadence == 0

            if merge_cadence == float('inf') and not is_last:
                continue

            if call_merger or (not call_merger and is_last):
                print('\n\nSending Chunks to Merger!\n\n')

                # Create a dictionary without chunk_id for async_merge_chunks
                chunk_summaries_no_id = {key: value["summary"] for key, value in chunk_summaries.items()}

                # Print chunk_ids being processed
                processing_chunk_ids = [value["chunk_id"] for value in chunk_summaries.values()]
                print(f"Processing chunk_ids: {processing_chunk_ids}")

                merge_thread = threading.Thread(
                    target=async_merge_chunks,
                    args=(chunk_summaries_no_id, merge_start_time, doc.metadata['end_time'],
                        args.outfile, args.extend_to_vertex, cloud_model, cloud_prompt,
                        args.anomaly_thresh, loader, doc, mode, processing_chunk_ids, merge_threads)
                )
                merge_thread.start()
                merge_threads.append(merge_thread)

                if mode == "w":
                    mode = "a"
                merge_start_time = doc.metadata['end_time'] - args.chunk_overlap

                # Remove processed chunks from chunk_summaries
                print(f"Cleaning processed chunk_ids: {processing_chunk_ids}")
                chunk_summaries = {key: value for key, value in chunk_summaries.items()
                                if value["chunk_id"] not in processing_chunk_ids}

    for t in merge_threads:
        t.join()

    if ingest_demo_chunks:
        while not ingest_queue.empty():
            time.sleep(0.5)
        ingest_queue.put("END")

    print("\nTotal Inference Time: {} sec\n".format(time.time() - tot_inf_st_time))

    return all_chunk_outputs  

if __name__ == '__main__':
    parser_txt = "Generate video summarization using LangChain, OpenVINO-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str, help='Path to video you want to summarize.')
    parser.add_argument("model_dir", type=str, help="Path to openvino-genai optimized model")
    parser.add_argument("-p", "--prompt", type=str, default="Please summarize this video.")
    parser.add_argument("-d", "--device", type=str, default="CPU")
    parser.add_argument("-t", "--max_new_tokens", type=int, default=256)
    parser.add_argument("-f", "--max_num_frames", type=int, default=64)
    parser.add_argument("-c", "--chunk_duration", type=int, default=15)
    parser.add_argument("-v", "--chunk_overlap", type=int, default=2)
    parser.add_argument("-mc", "--merge_cadence", type=int, default=30)
    parser.add_argument("-r", "--resolution", type=int, nargs=2)
    parser.add_argument("-o", "--outfile", type=str, default='')
    parser.add_argument("-e", "--extend_to_vertex", action="store_true", default=True)
    parser.add_argument("-a", "--anomaly_thresh", type=float, default=0.0)
    parser.add_argument("-m", "--cloud_model", type=str, default="gemini-2.0-flash-exp")
    
    args = parser.parse_args()
    summarizer_main(args)
