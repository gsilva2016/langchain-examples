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
import time
from langchain_core.documents import Document
from langchain_core.load import dumpd, dumps, loads
#import streamlit_scrollable_textbox as stx

load_dotenv() # This loads the variables from .env into the environment


# Whisper API URL
#WHISPER_API_URL = "http://127.0.0.1:5910/inference"
OVMS_ENDPOINT = os.getenv("OVMS_ENDPOINT", "http://localhost:8013/v3/chat/completions")
OVMS_MODEL = os.getenv("OVMS_MODEL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")


st.set_page_config(page_title="Diarized Sentiment via Audio Speech Recognition", layout="wide", initial_sidebar_state="auto")

# Initialize session state
if 'transcription' not in st.session_state:
   st.session_state.transcription = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'json_data' not in st.session_state:
    st.session_state.json_data = {}
if 'confirmed_data' not in st.session_state:
    st.session_state.confirmed_data = {}
if 'show_editor' not in st.session_state:
    st.session_state.show_editor = False

def transcribe_audio(audio_path):
    """Send recorded audio to Whisper API and return the transcription."""
    docs = None
    try:        
        #loader = loadWordLevelASR("Qwen/Qwen2-1.5B-Instruct", "cpu")
        loader = loadWordLevelASR(WHISPER_MODEL, ASR_DEVICE)
        start_time = time.time()
        docs = loader(audio_path, batch_size=1 ,return_timestamps="word")

        if "xpu" in str(loader.device):
            torch.xpu.synchronize()

        print("ASR Latency: ", time.time() - start_time, flush=True)
        #response = requests.post(WHISPER_API_URL, files=files, data=data, proxies=proxies)
    except Exception as ex:
        print("ASR error: ", ex, flush=True)

#if not docs is None:
    return docs

def preprocess_audio(audio_path):
    """Preprocess recorded audio to 16kHz."""
    y, sr = librosa.load(audio_path, sr=16000)
    sf.write(audio_path, y, 16000)
    return audio_path

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

def diarizedSentiment(transcript, diarization):
    endpoint = OVMS_ENDPOINT
    model_id = OVMS_MODEL
    device = ""
    batch_size = 1
    max_tokens = 200

    sentimenter = loadSentimenter(endpoint, model_id, device, batch_size, max_tokens)
    total_start_time = time.time()
    # wait for other services to send data
    diarize = diarization
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
    
    #st.session_state.show_editor = True

    os.remove(temp_audio_path)

# for display the json and can be edit by user
def display_json_editor():
    if not st.session_state.json_data:
        return None
        
    data = st.session_state.json_data
    
    st.subheader("Edit Medical Information:")
    
    chief_complaint = st.text_input("Chief Complaint", value=data.get('chief_complaint', ''), key='chief_complaint')
    symptom_description = st.text_input("Symptom Description", value=data.get('symptom_description', ''), key='symptom_description')
    symptom_duration = st.text_input("Symptom Duration", value=data.get('symptom_duration', ''), key='symptom_duration')
    associated_symptoms = st.text_input("Associated Symptoms", value=str(data.get('associated_symptoms', '')), key='associated_symptoms')
    past_medical_history = st.text_input("Past Medical History", value=data.get('past_medical_history', ''), key='past_medical_history')
    current_medications = st.text_input("Current Medications", value=data.get('current_medications', ''), key='current_medications')
    medication_allergies = st.text_input("Medication Allergies", value=data.get('medication_allergies', ''), key='medication_allergies')
    surgical_history = st.text_input("Surgical History", value=data.get('surgical_history', ''), key='surgical_history')
    family_history = st.text_input("Family History", value=data.get('family_history', ''), key='family_history')
    smoking_status = st.text_input("Smoking Status", value=data.get('smoking_status', ''), key='smoking_status')
    alcohol_consumption = st.text_input("Alcohol Consumption", value=data.get('alcohol_consumption', ''), key='alcohol_consumption')

    if st.button('Confirm Data', key='confirm_button'):
        st.session_state.confirmed_data = {
            'chief_complaint': chief_complaint,
            'symptom_description': symptom_description,
            'symptom_duration': symptom_duration,
            'associated_symptoms': associated_symptoms,
            'past_medical_history': past_medical_history,
            'current_medications': current_medications,
            'medication_allergies': medication_allergies,
            'surgical_history': surgical_history,
            'family_history': family_history,
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption
        }
        st.success("Data confirmed! You can now send to EMR.")
        st.json(st.session_state.confirmed_data)
        # st.rerun()

# Streamlit UI
st.title("Diarized Sentiment via Audio Speech Recognition")
st.write("Upload or record audio, and diarize the transcription with sentiment analysis using Whisper and Qwen2 SLMs")
    

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
        handle_audio_submission(temp_audio_path)

if uploaded_file is not None:
    if "wav" in uploaded_file:
        ft = ".wav"
    else:
        ft = ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ft) as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        temp_audio_path = temp_audio.name
    
    if st.sidebar.button("Submit Query", key='submit_uploaded'):
        handle_audio_submission(temp_audio_path)

# Show JSON editor if data is available
#if st.session_state.show_editor and st.session_state.json_data:
#    display_json_editor()
