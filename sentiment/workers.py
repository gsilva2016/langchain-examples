from dotenv import load_dotenv
import torch
from langchain_core.documents import Document
from langchain_core.load import dumpd, dumps, loads
from typing import List
import numpy as np
import sys
import time
from multiprocessing import Queue
from utils import *


def performASR(q: Queue, audio_ptr, model_id, asr_device, batch_size=1):
    try:
        loader = loadWordLevelASR(model_id, asr_device)
    except Exception as ex:
        print("ASR error: ", ex, flush=True)
        return

#    asr_docs = loads(asr_loader.load()) if args.asr_endpoint != "" else asr_loader.load()
    start_time = time.time()
    try:
        docs = loader(audio_ptr, batch_size=batch_size ,return_timestamps="word")
    except Exception as ex:
        print("ASR processing error: ", ex)

    if "xpu" in str(loader.device):
        torch.xpu.synchronize()

    q.put(docs)
    print("ASR Latency: ", time.time() - start_time, flush=True)

def performDiarize(q: Queue, endpoint, audio_file, model_id):
    try:
        loader = loadDiarizer(endpoint, audio_file, model_id)
    except Exception as ex:
        print("Diarization error: ", ex, flush=True)
        return

    start_time = time.time()
    try:
        docs = loads(loader.load()) if loader.api_base != None else loader.load()
    except Exception as ex:
        print("Diarization processing error: ", ex)

    print("Diarization Latency: ", time.time() - start_time, flush=True)
    q.put(docs)


def performDiarizedSentiment(sentiment_q: Queue, transcript_q: Queue, diarize_q: Queue, endpoint, model_id, device, max_tokens, batch_size =1): #-> List[Document]:
    
    sentimenter = loadSentimenter(endpoint, model_id, device, batch_size, max_tokens)
    total_start_time = time.time()
    # wait for other services to send data
    transcript = transcript_q.get()
    diarize = diarize_q.get()
    end_timestamps = []

    start_time = time.time()
    for i,chunk in enumerate(transcript['chunks']):
        #print(chunk)
        ts = eval(str(chunk["timestamp"]))
        if ts is None:
            end_timestamps.append(sys.float_info.max)
        else:
            end_timestamps.append(ts[1])
    end_timestamps = np.array(end_timestamps)

    #  Make speakers unique. Reference Credit (Apache 2.0): https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
    new_segs = []
    prev_seg = cur_seg = diarize[0]
    for i in range(1,len(diarize)):
        cur_seg = diarize[i]
        if cur_seg.metadata["speaker"] != prev_seg.metadata["speaker"] and i < len(diarize):
            new_segs.append(
                Document(
                    page_content=f"start={prev_seg.metadata['start']}s stop={cur_seg.metadata['start']}s {prev_seg.metadata['speaker']}",
                    metadata={
                        "speaker": prev_seg.metadata['speaker'],
                        "start": f"{prev_seg.metadata['start']}",
                        "stop": f"{cur_seg.metadata['start']}",
                    }
                )
            )
            prev_seg = diarize[i]

    new_segs.append(
        Document(
            page_content=f"start={prev_seg.metadata['start']}s stop={cur_seg.metadata['start']}s {prev_seg.metadata['speaker']}",
            metadata={
                "speaker": prev_seg.metadata['speaker'],
                "start": f"{prev_seg.metadata['start']}",
                "stop": "9999", # Give rest of text for last speaker regardless of diarizer end_time
            }
        )
    )

    segmented_preds = []
    transcript = transcript['chunks']

    for segment in new_segs:
        end_time = eval(segment.metadata["stop"])
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        chunk_text = ""
        for i in range(upto_idx + 1):
            chunk_text += transcript[i]['text']

        if endpoint == "":
            prompt = getQwenTemplateLocal(chunk_text)
        else:
            prompt = getQwenTemplateRemote(chunk_text)

        sentiment_result = sentimenter.invoke(prompt).content if endpoint != "" else sentimenter.invoke(prompt)
        segmented_preds.append( Document(
            page_content=f"A {sentiment_result} speaker: {segment.metadata['speaker']} stated: {chunk_text}",
            metadata = {
                "speaker": segment.metadata['speaker'],
                "text": chunk_text,
                "text-sentiment": sentiment_result
            }
        ))

        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break
    print("Diarized Sentiment Latency: ", time.time() - start_time, flush=True)
    print("Total Pipeline Latency: ", time.time() - total_start_time, flush=True)
    sentiment_q.put(segmented_preds)
