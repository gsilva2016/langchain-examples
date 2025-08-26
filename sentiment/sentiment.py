import argparse
import time
import os
from langchain_core.documents import Document
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor 
from multiprocessing import Manager
from typing import List
from workers import performASR, performDiarize, performDiarizedSentiment
from utils import *

MAX_NEW_TOKENS = 200
qmanager = Manager()
asr_q = qmanager.Queue()
diarize_q = qmanager.Queue()
sentiment_q = qmanager.Queue()

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--asr_endpoint", nargs="?", default="")
parser.add_argument("--diarize_endpoint", nargs="?", default="")
parser.add_argument("--sentiment_endpoint", nargs="?", default="")
parser.add_argument("--asr_model_id", nargs="?", default="openai/whisper-tiny") #default="distil-whisper/distil-small.en")
parser.add_argument("--diarize_model_id", nargs="?", default="pyannote/speaker-diarization-3.1")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--asr_batch_size", default=1, type=int)
parser.add_argument("--asr_load_in_8bit", default=False, action="store_true")
parser.add_argument("--sentiment_model_id", nargs="?", default="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4")
parser.add_argument("--demo_mode", default=False, action="store_true")
args = parser.parse_args()

print("LangChain OpenVINO Sentiment Analysis")
print("ASR model_id: ", args.asr_model_id)
print("ASR batch_size: ", args.asr_batch_size)
print("ASR load_in_8bit: ", args.asr_load_in_8bit)
print("Diarize model_id: ", args.diarize_model_id)
print("Sentiment model_id: ", args.sentiment_model_id)
print("Audio file: ", args.audio_file)
print("Demo Mode Enabled: ", args.demo_mode)
print("ASR Endpoint: ", args.asr_endpoint)
print("Diarize Endpoint: ", args.diarize_endpoint)
print("Sentiment Endpoint: ", args.sentiment_endpoint)
print("ASR Device: ", args.device)

asr_device = args.device #"GPU" if "GPU" in args.device else args.device
model_id = args.asr_model_id # openai/whisper-tiny

check_audio_file = Path(args.audio_file)
if args.demo_mode and not check_audio_file.exists():
    # create empty file so ASR can initialize
    check_audio_file.touch()
if args.asr_endpoint != "":
    raise Exception("Word-level ASR endpoint not supported.")

futures = []
with ProcessPoolExecutor() as pool:

    diarized_sentiment_future = pool.submit(
        performDiarizedSentiment,
        sentiment_q,
        asr_q,
        diarize_q,
        args.sentiment_endpoint,
        args.sentiment_model_id,
        "cpu",
        MAX_NEW_TOKENS,
        1
    )
    futures.append(diarized_sentiment_future)

    diarize_future = pool.submit(
        performDiarize,
        diarize_q,
        args.diarize_endpoint,
        args.audio_file,
        args.diarize_model_id
    )
    futures.append(diarize_future)

    asr_future = pool.submit(
        performASR,
        #asr_loader,
        asr_q,
        args.audio_file,
        model_id,
        asr_device
    )
    futures.append(asr_future)



print("Initialization completed...\n")

while True:

    sentiment = sentiment_q.get()

    for i,d in enumerate(sentiment):
        print(d.page_content)

    if not args.demo_mode:
        break
    else:
        time.sleep(2)
