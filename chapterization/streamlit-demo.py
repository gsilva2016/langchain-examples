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
import time
import gc
import docs_loader_utils as docs_loader
from langchain_core.documents import Document

# Replace with whispercpp server?
from langchain_openvino_asr import OpenVINOSpeechToTextLoader

# Replace with OVMS
from langchain_huggingface import HuggingFacePipeline

#WHISPER_API_URL = "http://127.0.0.1:5910/inference"
#LLM_API_URL = "http://localhost:8013/v3/chat/completions"
MAX_NEW_TOKENS = 12000

class WorkerResult:
    def __init__(self, latency, data):
        self.latency = latency
        self.data = data

st.set_page_config(page_title="Chapterization Demo", layout="wide", initial_sidebar_state="auto")

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""

def transcribe_audio(audio_path):
    """Send recorded audio to Whisper API and return the transcription."""
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"response-format": "json"}
        proxies = {"http": None, "https": None}
        
        # init
        asr_loader = OpenVINOSpeechToTextLoader(
            audio_path,
            ASR_MODEL,
            device=INF_DEVICE,
            load_in_8bit=ASR_LOAD_IN_8BIIT,
            batch_size=ASR_BATCH_SIZE
        )

        start_time = time.time()
        docs =  asr_loader.load()
        print("ASR latency: ", time.time() - start_time)

    # clean memory
    asr_loader = None
    collected = gc.collect()

    # Chunk with timestamps in mind
    docs_big = docs_loader.chunk_transcript_docs(docs, chunk_size=1500)
        
    # chunk orig transcript by timestamp. Used for mapping big chunk back to original for more fidelity
    docs = docs_loader.chunk_transcript_docs(docs, chunk_size=5)

    if NUM_CLUSTERS < 1:
        docs_big = [Document(page_content=docs_loader.format_transcript_docs_sectionize_big(docs_big, docs))]
    else:
        from langchain_community.embeddings import OpenVINOEmbeddings
        model_name = "sentence-transformers/all-mpnet-base-v2" # "sentence-transformers/all-MiniLM-L6-v2" # "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": INF_DEVICE}
        encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}
        ov_embeddings = OpenVINOEmbeddings(        model_name_or_path=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        # Convert to most relevant text in a single doc for single inference
        import numpy as np
        from sklearn.cluster import KMeans

        # 768 dimensional
        vectors = ov_embeddings.embed_documents([x.page_content for x in docs_big])
        print("KMeans fit proceeding...")
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        kmeans.fit(vectors)

        closest_indices = []

        # Loop through the number of clusters you have
        for i in range(NUM_CLUSTERS):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        selected_indices=sorted(closest_indices)
        last_idx_sel = selected_indices[len(selected_indices)-1]
        last_idx_docs = len(docs_big) - 1
        if last_idx_sel != last_idx_docs:
            # change final end time to end of stream
            docs_big[last_idx_sel].metadata['end'] = docs_big[last_idx_docs].metadata['end']
        docs_big = [docs_big[doc] for doc in selected_indices]
        docs_big = [Document(page_content=docs_loader.format_transcript_docs_sectionize_big(docs_big, docs))]
        print("KMeans fit completed")
    return docs_big

        #response = requests.post(WHISPER_API_URL, files=files, data=data, proxies=proxies)
        #print("ASR Time: ", time.time() - start)
        #print("DEBUG: ", response)
        
    #if response.status_code == 200:
    #    return response.json().get("text", "No transcription received.")
    #else:
    #    return f"Error: {response.status_code} - {response.text}"

def summarizer(transcript):
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache-ov-models"}
    print("DEBUG: ", LLM_MODEL, LLM_BATCH_SIZE, INF_DEVICE, MAX_NEW_TOKENS, LLM_TEMPERATURE)
    ov_llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text-generation",
        backend="openvino",
        batch_size=LLM_BATCH_SIZE,
        model_kwargs={"device": INF_DEVICE, "ov_config": ov_config, "trust_remote_code": True,},
        pipeline_kwargs={
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "top_k": 10,
            "temperature": LLM_TEMPERATURE,
            "return_full_text": False,
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
            "use_cache": True
        }
    )
    ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id
    tokenizer = ov_llm.pipeline.tokenizer
    print("Starting chapterization...")


    for i in range(0, len(transcript)):
        doc_in = [transcript[i]]
        text = docs_loader.format_docs(docs_loader.format_docs_ts(doc_in))
        print("Transcript for summary: ", text)

        # Identify and summarize the five most critical points discussed in this
        template = [
        {"role": "user", "content": f"""Instruction:\nAnalyze the provided educational transcript in JSON format and generate a chapterization in chronological order for the provided list of text segments. Review the list of "text" segments by combining them into a single text to identify the 5 most critical topics and at which time those topics were discussed to create the chapter summaries. For every chapter summary created use the first text segment's start time and the last text segment's end time for the topic being summarized.  Be sure the 5 topics reflect the summarizations and not a generic chapter title such as "Conclusion" or "Introduction". The chapter summarizations must comprehend the entire text, be organized with correct spelling, and use the following example format as a template only:
topic: "Chapter 1 - Introduction" \n
start: "00:00:00.000" \n
end: "00:04:02.541" \n
summary: "..."\nChapterize: {text}""" }]

        template = [
        {"role": "system", "content": "You analyze a provided text and identify the 5 most critical topics to summarize from it. All information generated by you is factual from the provided text. All start and end times must be from the provided text."},
        {"role": "user", "content": f"""Instruction:\nAnalyze the provided educational transcript in JSON format and generate a chapterization in chronological order for the provided list of text segments. Review the list of "text" segments by combining them into a single text to identify the 5 most critical topics and at which time those topics were discussed to create the chapter summaries. For every chapterizaton identified use the first text segment's start time and the last text segment's end time for the topic being summarized. Use only text and timestamps from the provided text segments.  The 5 topic titles must reflect the summarizations and not be a generic topic title such as "Conclusion" or "Introduction". The chapter summarizations must comprehend the entire text, be organized with correct spelling, and use the following example format as a template only:
topic: "Chapter 1 - Introduction" \n
start: "hh:mm:ss.ms" \n
end: "hh:mm:ss.ms" \n
summary: "..."\nChapterize: {text}""" }]

        template = [
#        {"role": "system", "content": "All information generated by you must be from the provided text. All start and end times must be from the provided text."},
        {"role": "system", "content": "Use only start and end timestamps from the provided text segments."},
        {"role": "user", "content": f"""Instruction:\nAnalyze the provided educational transcript in JSON format and generate a chapterization in chronological order for the provided list of text segments. Review the list of "text" segments by combining them into a single text to identify the 5 most critical topics and at which time those topics were discussed to create the chapter summaries. For every chapterizaton identified use the first text segment's start time and the last text segment's end time for the topic being summarized. Use only text and timestamps from the provided text segments.  The 5 topic titles must reflect the summarizations and not be a generic topic title such as "Conclusion" or "Introduction". The chapter summarizations must comprehend the entire text, be organized with correct spelling, and use the following example format as a template only:
topic: "Chapter 1 - Introduction" \n
start: "00:00:00.000" \n
end: "00:04:02.541" \n
summary: "..."\nChapterize: {text}""" }]

        print("DEBUG: ", template)

        formatted_prompt = ov_llm.pipeline.tokenizer.apply_chat_template(template, tokenize=False)
        tokens = tokenizer.tokenize(formatted_prompt)
        num_tokens = len(tokens)
        print(f"Number of input tokens: {num_tokens}\n")

        summary = ov_llm.invoke(formatted_prompt)
        summary = summary.replace('assistant\n\n', '')
        transcript[i].metadata["summary"] = summary
        return summary

def preprocess_audio(audio_path):
    """Preprocess recorded audio to 16kHz."""
    y, sr = librosa.load(audio_path, sr=16000)
    sf.write(audio_path, y, 16000)
    return audio_path

def handle_audio_submission(temp_audio_path):
    temp_audio_path = preprocess_audio(temp_audio_path)
    st.session_state.transcription = transcribe_audio(temp_audio_path)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("Transcription:")
        with st.container(border=True):
            st.write(st.session_state.transcription)
        
    with col2:
        st.subheader("Chapterization")
        with st.container(border=True):
            st.session_state.summary = summarizer(st.session_state.transcription)
            st.write(st.session_state.summary)
    os.remove(temp_audio_path)

# Streamlit UI
st.title("Chapterization Demo")
st.write("Chapterize uploaded or recorded audio.")

ASR_MODEL = "openai/whisper-small.en"
ASR_MODEL = st.sidebar.selectbox(
        "ASR Model: ",
        [
            ASR_MODEL,
            "openai/whisper-medium.en", 
            "distil-whisper/distil-small.en"
        ]
)

LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLM_MODEL = st.sidebar.selectbox(
        "Chapterizer Model: ",
        [
            LLM_MODEL,  
            "Qwen/Qwen2.5-3B", 
            "llmware/llama-3.2-3b-instruct-ov"
        ]
)

NUM_CLUSTERS = 3
NUM_CLUSTERS = st.sidebar.selectbox(
        "Number K-mean clusters: ",
        [
            NUM_CLUSTERS, 0, 1, 2, 4, 5, 6, 7, 8 ,9, 10, 11, 12
        ]
)


ASR_LOAD_IN_8BIIT = st.sidebar.radio(
        "ASR 8BIT: ",
        [
            "True", "False"
        ]
)

ASR_BATCH_SIZE = 1
ASR_BATCH_SIZE = st.sidebar.selectbox(
        "ASR Batch Size: ",
        [
            ASR_BATCH_SIZE, 2
        ]
)

INF_DEVICE = "GPU"
INF_DEVICE = st.sidebar.radio(
        "Inference Device: ",
        [
            INF_DEVICE, "CPU"
        ]
)

LLM_BATCH_SIZE = 2
LLM_BATCH_SIZE = st.sidebar.selectbox(
        "LLM Batch Size: ",
        [
            LLM_BATCH_SIZE, 1 
        ]
)

LLM_TEMPERATURE = .02
LLM_TEMPERATURE = st.sidebar.selectbox(
        "LLM Temperature: ",
        [
            LLM_TEMPERATURE, .5, 0.7, 1.0, 1.2
        ]
)


# Upload audio feature
uploaded_file = st.sidebar.file_uploader("Upload a WAV audio file", type=["wav"])

# Audio recording feature
audio_data = st.sidebar.audio_input("Record Audio")


# Handle audio submission/begin of pipeline
if audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data.getvalue())
        temp_audio_path = temp_audio.name

    st.sidebar.success(f"Audio recorded and saved to {temp_audio_path}")

    if st.sidebar.button("Submit Query", key='submit_recorded'):
        handle_audio_submission(temp_audio_path)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        temp_audio_path = temp_audio.name
    
    if st.sidebar.button("Submit Query", key='submit_uploaded'):
        handle_audio_submission(temp_audio_path)
