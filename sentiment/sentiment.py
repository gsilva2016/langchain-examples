import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.documents import Document
from pathlib import Path

from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader
from langchain_openai import ChatOpenAI # Use with OVMS
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
from langchain_core.load import dumpd, dumps, loads
import numpy as np
import asyncio


MAX_NEW_TOKENS = 200
asr_docs = []
diarize_docs = []

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--asr_endpoint", nargs="?", default="")
parser.add_argument("--diarize_endpoint", nargs="?", default="")
parser.add_argument("--sentiment_endpoint", nargs="?", default="")
parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
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
print("Inference device  : ", args.device)
print("Audio file: ", args.audio_file)
print("Demo Mode Enabled: ", args.demo_mode)
print("ASR Endpoint: ", args.asr_endpoint)
print("Diarize Endpoint: ", args.diarize_endpoint)
print("Sentiment Endpoint: ", args.sentiment_endpoint)
#input("Press Enter to continue...")

def getQwentTemplateRemote(text: str):
    messages = [
    (
        "system",
        "You are trained to analyze and detect the sentiment of the given text. If you are unsure of an answer, you can say \"not sure\" and recommend the user review manually",
    ),
    ("user", f"Analyze the following text and determine if the sentiment is: Positive, Negative, or Neutral. {text}"
    )]
    return messages

def getQwenTemplateLocal(text: str):
    template = f"""<|im_start|>system\nYou are trained to analyze and detect the sentiment of the given text. If you are unsure of an answer, you can recommend the user review manually<|im_end|>\n<|im_start|>user\nIs the sentiment for the provided text '{text}' Positive, Negative, or Neutral?<|im_end|>\n<|im_start|>assistant\n"""

    return template

async def gather_asr_diarize():
    results = await asyncio.gather(performASR(), performDiarize())
    return results

async def performASR():
    asr_docs = loads(asr_loader.load()) if args.asr_endpoint != "" else asr_loader.load()
    return asr_docs

async def performDiarize():
    diarize_docs = loads(diarize_loader.load()) if args.diarize_endpoint != "" else diarize_loader.load()
    return diarize_docs

def diarized_sentiment(transcript, diarize):
    end_timestamps = []
    for i, chunk in enumerate(transcript):
        ts = eval(chunk.metadata["timestamp"])
        if ts is None:
            end_timestamps.append(sys.float_info.max)
        else:
            end_timestamps.append(ts[1])
    end_timestamps = np.array(end_timestamps)

    segmented_preds = []

    for segment in diarize:
        end_time = eval(segment.metadata["stop"])
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        for i in range(upto_idx + 1):
            if args.sentiment_endpoint == "":
                prompt = getQwenTemplateLocal(transcript[i].page_content)
            else:
                prompt = getQwenTemplateRemote(transcript[i].page_content)

            sentiment_result = ov_llm.invoke(prompt).content if args.sentiment_endpoint != "" else ov_llm.invoke(prompt)

            segmented_preds.append( Document(
                page_content=f"speaker: {segment.metadata['speaker']} stated: {transcript[i].page_content} with sentiment: {sentiment_result}",
                metadata = {
                    "speaker": segment.metadata['speaker'],
                    "text": transcript[i].page_content,
                    "text-sentiment": sentiment_result
                }   
            ))

        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break

    return segmented_preds



print("Loading ASR...")
check_audio_file = Path(args.audio_file)
if args.demo_mode and not check_audio_file.exists():
    # create empty file so ASR can initialize
    check_audio_file.touch()

asr_loader = OpenVINOSpeechToTextLoader(
    args.audio_file, 
    api_base = None if args.asr_endpoint == "" else args.asr_endpoint,
    model_id = args.asr_model_id, 
    device=args.device, 
    load_in_8bit=args.asr_load_in_8bit, 
    batch_size=args.asr_batch_size
)

print("Loading Diarize...")
diarize_loader = OpenVINOSpeechDiarizeLoader(
    api_base = None if args.diarize_endpoint == "" else args.diarize_endpoint,
    file_path = args.audio_file,
    model_id = args.diarize_model_id
)

print("Loading Sentiment...")
if args.sentiment_endpoint == "":
    print("Loading local sentiment...")
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY", 
        "NUM_STREAMS": "1", 
        "CACHE_DIR": "./cache-ov-model",
    }
    ov_llm = HuggingFacePipeline.from_model_id(
        model_id=args.sentiment_model_id,
        task="text-generation",
        backend="openvino",
        batch_size=1,
        model_kwargs={
            "device": args.device, 
            "ov_config": ov_config
        },
        pipeline_kwargs={
            "max_new_tokens": MAX_NEW_TOKENS,
            "return_full_text": False,
#            "repetition_penalty": 1.2,
#            "encoder_repetition_penalty": 1.2,
#            "top_p": 0.8,
            "temperature": 0.1,
        })
    ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id
else:
    print("Loading remote sentiment...")
    ov_llm = ChatOpenAI(
        model=args.sentiment_model_id,
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
        timeout=None,
        max_retries=2,
        api_key="provided_by_user_when_needed",  # if you prefer to pass api key in directly instaed of using env
        base_url=args.sentiment_endpoint,
    )

print("Initialization completed...\n")
input("Press Enter to continue...")
while True:
    start_time = time.time()
    try:
        results = asyncio.run(gather_asr_diarize())
    except Exception as exc:
        raise Exception(
                "Error asr/diarize loading."
        ) from exc

    sentiment = dumpd(diarized_sentiment(results[0], results[1]))
    end_time = time.time()
    print(sentiment)
    print("\n\n")
    print("Total time taken for completion: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")

    if not args.demo_mode:
        break
    else:
        time.sleep(2)
