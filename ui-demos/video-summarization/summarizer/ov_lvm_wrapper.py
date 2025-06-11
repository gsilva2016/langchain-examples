from typing import List, Optional

import openvino_genai
from decord import VideoReader, cpu
from langchain.llms.base import LLM
from openvino import Tensor
import queue
import time
import threading

# Create thread safe queue
stream_queue = queue.Queue()

# Create shared state variables for tracking chunk_id
last_token_time = None
chunk_id = 0
chunk_lock = threading.Lock()
new_chunk_flag = False
generation_started = False
is_first_token = True
append_newline = False
chunk_start = 0
chunk_duration = 15
chunk_overlap = 2
chunk_end = chunk_duration

def reset_chunk_variables():
    global last_token_time, chunk_id, new_chunk_flag, generation_started, is_first_token
    global chunk_start, chunk_end, chunk_duration, chunk_overlap, append_newline    
    with chunk_lock:       
        last_token_time = None       
        chunk_id = 0       
        new_chunk_flag = False       
        generation_started = False       
        is_first_token = True       
        append_newline = False      
        chunk_start = 0       
        chunk_end = chunk_duration

def reset_chunk_index():
    global chunk_id, chunk_start, chunk_end
    chunk_id = 0
    chunk_start = 0
    chunk_end = chunk_duration

def encode_video(video_path: str,
                 max_num_frames: int = 64,
                 resolution: list = []) -> list:
    def uniform_sample(l: list, n: int) -> list:
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if len(resolution) != 0:
        vr = VideoReader(video_path, width=resolution[0],
                         height=resolution[1], ctx=cpu(0))
    else:
        vr = VideoReader(video_path, ctx=cpu(0))

    frame_idx = [i for i in range(0, len(vr), max(1, int(len(vr) / max_num_frames)))]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()

    frames = [Tensor(v.astype('uint8')) for v in frames]
    print('Num frames sampled:', len(frames))
    return frames

def chunk_watcher():
    global last_token_time, chunk_id, new_chunk_flag, generation_started, append_newline
    global chunk_start, chunk_end, chunk_duration, chunk_overlap
    while True:
        time.sleep(0.5)
        with chunk_lock:
            if generation_started and last_token_time is not None:
                if time.time() - last_token_time > 2.0:
                    append_newline = True
                    chunk_id += 1
                    new_chunk_flag = True
                    last_token_time = None
                    chunk_start = max(0, chunk_end - chunk_overlap)
                    chunk_end = chunk_start + chunk_duration

# Start watcher thread once at beggining of app
watcher_thread = threading.Thread(target=chunk_watcher, daemon=True)
watcher_thread.start()

def streamer(subword: str)-> bool:
    global last_token_time, new_chunk_flag, generation_started, chunk_id, is_first_token, append_newline
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    with chunk_lock:
        if not generation_started:
            reset_chunk_index()  # Reset chunk index on new summarization start

        generation_started = True
        if chunk_id == 0:
            chunk_id = 1

        if is_first_token:
            new_chunk_flag = True
            is_first_token = False

        if new_chunk_flag:
            #chunk_id += 1
            new_chunk_flag = False
            timestamp = f"[{chunk_start}-{chunk_end}sec]"
            curr_token = f"[CHUNK {chunk_id}] {timestamp}\n{subword}"
            
        else:
            curr_token = subword
        
        if append_newline:
            stream_queue.put("\n\n")
            append_newline = False
    
        last_token_time = time.time()
    print(curr_token, end='', flush=True)
    stream_queue.put(curr_token)
    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".

class OVMiniCPMV26Wrapper(LLM):
    ovpipe: object
    generation_config: object
    max_num_frames: int
    resolution: list[int]

    @property
    def _llm_type(self) -> str:
        return "Custom OV MiniCPM-V-2_6"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:

        # Parse prompt
        video_fh, question = prompt.split(',', 1)

        # Process text only
        if video_fh == '':
            self.ovpipe.start_chat()
            generated_text = self.ovpipe.generate(question,
                                                  generation_config=self.generation_config,
                                                  streamer=streamer)

        # Process video and text
        else:
            frames = encode_video(video_fh, self.max_num_frames,
                                  resolution=self.resolution)
            self.ovpipe.start_chat()
            generated_text = self.ovpipe.generate(question,
                                                  images=frames,
                                                  generation_config=self.generation_config,
                                                  streamer=streamer)
        self.ovpipe.finish_chat()
        #stream_queue.put("\n")
        return str(generated_text)


def OVMiniCPMV26Worker(model_dir: str,
                       device: str,
                       max_new_tokens: int,
                       max_num_frames: int,
                       resolution: list[int]) -> object:
    # Start ov genai pipeline
    enable_compile_cache = dict()
    if "GPU" == device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "./cache/vlm_cache"

    pipe = openvino_genai.VLMPipeline(model_dir, device, **enable_compile_cache)

    # Set variables for inference 
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    # Wrap for langchain integration
    ovminicpm_wrapper = OVMiniCPMV26Wrapper(ovpipe=pipe,
                                            generation_config=config,
                                            max_num_frames=max_num_frames,
                                            resolution=resolution)
    return ovminicpm_wrapper
