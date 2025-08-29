import streamlit as st
import requests
import tempfile
import os
from io import BytesIO
import numpy as np
import librosa
import soundfile as sf
import re
import json
from datetime import date
from utils import *
from dotenv import load_dotenv
import sys
import time
from langchain_core.documents import Document
from typing import Iterator
from langchain_core.load import dumpd, dumps, loads
from concurrent.futures.thread import ThreadPoolExecutor

load_dotenv() # This loads the variables from .env into the environment


# Whisper API URL
#WHISPER_API_URL = "http://127.0.0.1:5910/inference"
OVMS_ENDPOINT = os.getenv("OVMS_ENDPOINT", "http://localhost:8013/v3/chat/completions")
OVMS_MODEL = os.getenv("OVMS_MODEL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_ENDPOINT = os.getenv("ASR_ENDPOINT", "")

class WorkerResult:
    def __init__(self, latency, data):
        self.latency = latency
        self.data = data

st.set_page_config(page_title="Diarized Sentiment via Audio Speech Recognition", layout="wide", initial_sidebar_state="auto")

# Initialize session state
if 'transcription' not in st.session_state:
   st.session_state.transcription = ""
if 'diarize' not in st.session_state:
    st.session_state.diarize = ""
if 'diarizedsentiment' not in st.session_state:
    st.session_state.diarizedsentiment = ""
if 'asrTime' not in st.session_state:
    st.session_state.asrTime = ""
if 'diarizedTime' not in st.session_state:
    st.session_state.diarizedTime = ""
if 'diarizedsentimentTime' not in st.session_state:
    st.session_state.diarizedsentimentTime = ""

def transcribe_audio_t(audio_path):
    start = time.time()
    res = transcribe_audio(audio_path, CHUNK_LEN_S)
    return WorkerResult(time.time() - start, res)

def transcribe_audio_local(audio_path, chunk_len_s):
    loader = loadWordLevelASR(WHISPER_MODEL, ASR_DEVICE)

    if chunk_len_s != -1:
        # TODO: Batch =3 is a bit faster. tweaking strides/chunk_len give diff accuracy
        # stride_length_s=(10, 10)
        docs = loader(audio_path, batch_size=3,return_timestamps="word", chunk_length_s=chunk_len_s, generate_kwargs={"language": "english"})
    else:
        docs = loader(audio_path, batch_size=1 ,return_timestamps="word", generate_kwargs={"language": "english"})

    if "xpu" in str(loader.device):
        torch.xpu.synchronize()

    return docs

def convert_to_docs(jsonObj) -> Iterator[Document]:
    #print("DEBUG: " , jsonObj)
    #jsonObj = json.loads(jsonStr)
    docs = '{ "chunks": ['
    segs = jsonObj["segments"]
    for seg in segs:
        docs = docs + '{"text":"' + seg["text"].replace('"', '\\"') + '", "timestamp":[' 
        if seg["start"] == "":
            print("ERROR!!!")
            return None
        if seg["end"] == "":
            print("ERR:::")
            return None
        docs = docs + str(seg["start"]) + ',' + str(seg["end"]) + ']},'
        #docs.append(Document(
        #    page_content=seg["text"],
        #    metadata={
        #        "timestamp": "(" + str(seg["start"]) + ", " + str(seg["end"]) + ")"
        #    }
        #))
    docs = docs[:-1] + ']}'

    with open('largefile.txt', 'w') as f:
        f.write(docs)


    print("DEBUG: " , docs)

    return json.loads(docs)

def transcribe_audio_endpoint(audio_path):
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"max_len": 1, "response_format": "verbose_json"}
        proxies = {"http": None, "https": None}
        start = time.time()
        response = requests.post(
            ASR_ENDPOINT, 
            files=files, 
            data=data, 
            proxies=proxies, 
            stream=False
        )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    return convert_to_docs(response.json())

def transcribe_audio(audio_path, chunk_len_s=-1):
    """Send recorded audio to Whisper API and return the transcription."""
    docs = None
    try:
        audio_path = preprocess_audio(audio_path)
        start_time = time.time()
        print("DEBUG: endpoint is  ", ASR_ENDPOINT)
        if ASR_ENDPOINT == "":
            docs = transcribe_audio_local(audio_path, chunk_len_s)
        else:
            docs = transcribe_audio_endpoint(audio_path)

        print("ASR Latency: ", time.time() - start_time, flush=True)
        #response = requests.post(WHISPER_API_URL, files=files, data=data, proxies=proxies)
    except Exception as ex:
        print("ASR error: ", ex, flush=True)

    return docs

def preprocess_audio(audio_path):
    """Preprocess recorded audio to 16kHz."""
    y, sr = librosa.load(audio_path, sr=16000)
    sf.write(audio_path, y, 16000)
    return audio_path

def diarizer_t(audio_file):
    start = time.time()
    res = diarizer(audio_file)
    return WorkerResult(time.time() - start, res)

def diarizer(audio_file):
    model_id = DIARIZE_MODEL
    endpoint = ""

    try:
        
        loader = loadDiarizer(endpoint, audio_file, model_id)
    except Exception as ex:
        print("Diarization error: ", ex, flush=True)
        return None

    start_time = time.time()
    try:
        docs = loads(loader.load()) if loader.api_base != None else loader.load()
    except Exception as ex:
        print("Diarization processing error: ", ex)

    print("Diarization Latency: ", time.time() - start_time, flush=True)
    return docs


def diarizedSentiment_t(transcript, diarization, enable_sentiment):
    start = time.time()
    res = diarizedSentiment(transcript, diarization, enable_sentiment)
    return WorkerResult(time.time() - start, res)

def diarizedSentiment(transcript, diarization, enable_sentiment = True):
    endpoint = OVMS_ENDPOINT
    model_id = OVMS_MODEL
    device = ""
    batch_size = 1
    max_tokens = 200

    if enable_sentiment:
        sentimenter = loadSentimenter(endpoint, model_id, device, batch_size, max_tokens)
    total_start_time = time.time()
    # wait for other services to send data
    diarize = diarization
    end_timestamps = []

    start_time = time.time()
    
    for i,chunk in enumerate(transcript['chunks']):
        #print(chunk)
        ts = eval(str(chunk["timestamp"]))
        if ts is None or ts[1] is None:
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

        if enable_sentiment:
            if endpoint == "":
                prompt = getQwenTemplateLocal(chunk_text)
            else:
                prompt = getQwenTemplateRemote(chunk_text)

            sentiment_result = sentimenter.invoke(prompt).content if endpoint != "" else sentimenter.invoke(prompt)
        else:
            sentiment_result = "N/A"

        segmented_preds.append( Document(
            page_content=f"Speaker: {segment.metadata['speaker']} | Stated: {chunk_text} | Emotion: {sentiment_result}",
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
    return segmented_preds

def beautifyDiarization(diarize_docs):
    output = ""
    for i,d in enumerate(diarize_docs):
        output = output + d.page_content
        output = output + "\n\n"
    return output

def startThreads(audio_path):
    futures = []
    with ThreadPoolExecutor() as pool:
        start = time.time()
        t1 = pool.submit(
            transcribe_audio_t,
            audio_path
        )
        t2 = pool.submit(
            diarizer_t,
            audio_path
        )
        t1res = t1.result()
        t2res = t2.result()
        st.session_state.transcription = t1res.data
        st.session_state.asrTime = t1res.latency

        st.session_state.diarize = t2res.data
        st.session_state.diarizeTime = t2res.latency

    res = diarizedSentiment_t(
        st.session_state.transcription,
        st.session_state.diarize,
        enable_sentiment
    )
    st.session_state.diarizedsentimentTime = res.latency
    st.session_state.diarizedsentiment = res.data
    total_time = time.time() - start
    updateUI(total_time)


def updateUI(total_time):
    st.subheader(f"Total Pipeline Latency: {total_time}")

    col1, col2 = st.columns(2)
    with col1: 
        st.subheader("Transcription:")
        with st.container(border=True, height=250):
            st.write(f"Latency: {st.session_state.asrTime} seconds")
            st.write(st.session_state.transcription)

    with col2:
        st.subheader("Diarization:")
        with st.container(border=True, height=250):
            st.write(f"Latency: {st.session_state.diarizeTime} seconds")
            st.write(st.session_state.diarize)

    col1t, col2t = st.columns(2)
    with col1t:
        if enable_sentiment:
            st.subheader("JSON Diarized Sentiment via ASR:")
        else:
            st.subheader("JSON Diarized ASR")

        with st.container(border=True, height=250):
            st.write(f"Latency: {st.session_state.diarizedsentimentTime} seconds")
            st.write(st.session_state.diarizedsentiment)
    with col2t:
        if enable_sentiment:
            st.subheader("Diarized Sentiment via ASR: ")
        else:
            st.subheader("Diarized ASR")

        with st.container(border=True, height=250):
            st.write(beautifyDiarization(st.session_state.diarizedsentiment))

def handle_audio_submission(temp_audio_path):
    temp_audio_path = preprocess_audio(temp_audio_path)
    st.session_state.transcription = transcribe_audio(temp_audio_path)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("Transcription:")
        with st.container(border=True, height=250):
            st.write(st.session_state.transcription)

    with col2:
        st.subheader("Diarization:")
        with st.container(border=True, height=250):
            st.session_state.diarize = diarizer(temp_audio_path)
            st.write(st.session_state.diarize)

    col1t, col2t = st.columns(2)
    with col1t:
        st.subheader("JSON Diarized Sentiment via ASR:")
        with st.container(border=True, height=250):
            st.session_state.diarizedsentiment = diarizedSentiment(st.session_state.transcription, st.session_state.diarize)
            st.write(st.session_state.diarizedsentiment)
    with col2t:
        st.subheader("Diarized Sentiment via ASR: ")
        with st.container(border=True, height=250):
            st.write(beautifyDiarization(st.session_state.diarizedsentiment))
    

    os.remove(temp_audio_path)

# Streamlit UI
st.title("Diarized Sentiment via Audio Speech Recognition")
st.write("Upload or record audio, and diarize the transcription with sentiment analysis using Whisper, Pyannote, and Qwen2 SLMs")


st.sidebar.write(f"**Diarize Model:** {DIARIZE_MODEL}")

if ASR_ENDPOINT:
    st.sidebar.write(f"**ASR Endpoint:** {ASR_ENDPOINT}")
else:
    WHISPER_MODEL = st.sidebar.selectbox(
        "ASR Model: ",
        [WHISPER_MODEL, "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3-turbo"]
    )
    CHUNK_LEN_S = st.sidebar.selectbox(
        "CHUNK_LEN_S",
        [1, 5, 10, 15, 25, 30 ]
    )


enable_sentiment = st.sidebar.checkbox("Enable Sentiment")

if enable_sentiment:
    st.sidebar.write(f"**Sentiment Model:** {OVMS_MODEL}")
    st.sidebar.write(f"**Sentiment API Endpoint:** {OVMS_ENDPOINT}")

# Upload audio feature
uploaded_file = st.sidebar.file_uploader("Upload a WAV/MP3 audio file", type=["wav", "mp3"])

# Audio recording feature
audio_data = st.sidebar.audio_input("Record Audio")

if audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data.getvalue())
        temp_audio_path = temp_audio.name

    st.sidebar.success(f"Audio recorded and saved to {temp_audio_path}")

    if st.sidebar.button("Submit Query", key='submit_recorded'):
        startThreads(temp_audio_path)
        #handle_audio_submission(temp_audio_path)

if uploaded_file is not None:
    if "wav" in uploaded_file:
        ft = ".wav"
    else:
        ft = ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ft) as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        temp_audio_path = temp_audio.name
    
    if st.sidebar.button("Submit Query", key='submit_uploaded'):
        startThreads(temp_audio_path)
        #handle_audio_submission(temp_audio_path)
