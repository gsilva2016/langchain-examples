import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
from langchain_openai_tts import OpenAIText2SpeechTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import docs_loader_utils as docs_loader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenVINOEmbeddings
import docs_loader_utils as docs_loader
import ast
import sounddevice as sd
import soundfile as sf
from pathlib import Path

MAX_NEW_TOKENS = 200

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--model_id", nargs="?", default="llmware/qwen2-0.5b-chat-ov")
parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
parser.add_argument("--tts_model_id", nargs="?", default="kokoro")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--asr_batch_size", default=1, type=int)
parser.add_argument("--asr_load_in_8bit", default=False, action="store_true")
parser.add_argument("--llm_batch_size", default=1, type=int)
parser.add_argument("--demo_mode", default=False, action="store_true")
parser.add_argument("--rag", default=False, action="store_true")
args = parser.parse_args()

print("LangChain OpenVINO Audio QnA")
print("LLM model_id: ", args.model_id)
print("LLM batch_size: ", args.llm_batch_size)
print("ASR model_id: ", args.asr_model_id)
print("ASR batch_size: ", args.asr_batch_size)
print("ASR load_in_8bit: ", args.asr_load_in_8bit)
print("TTS model_id: ", args.tts_model_id)
print("Inference device  : ", args.device)
print("Audio file: ", args.audio_file)
print("TTS model_id: ", args.tts_model_id)
print("Demo Mode Enabled: ", args.demo_mode)
print("RAG Enabled: ", args.rag)
#input("Press Enter to continue...")

if args.rag:
    print("Loading embeddings model...")
    embeddings = OpenVINOEmbeddings(
        model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {"device": args.device},
        encode_kwargs = {
            "mean_pooling": True,
            "normalize_embeddings": True
        }
    )

    print("Loading RAG data...")
    db = docs_loader.load_docs(embeddings, False)

print("Loading ASR...")
check_audio_file = Path(args.audio_file)
if args.demo_mode and not check_audio_file.exists():
    # create empty file so ASR can initialize
    check_audio_file.touch()

asr_loader = OpenVINOSpeechToTextLoader(args.audio_file, 
        args.asr_model_id, 
        device=args.device, 
        load_in_8bit=args.asr_load_in_8bit, 
        batch_size=args.asr_batch_size
)

print("Loading LLM...")
ov_config = {
        "PERFORMANCE_HINT": "LATENCY", 
        "NUM_STREAMS": "1", 
        "CACHE_DIR": "./cache-ov-model",
}
ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    batch_size=args.llm_batch_size,
    model_kwargs={
        "device": args.device, 
        "ov_config": ov_config
    },
    pipeline_kwargs={
        "max_new_tokens": MAX_NEW_TOKENS,
        "return_full_text": False,
        "repetition_penalty": 1.2,
        "encoder_repetition_penalty": 1.2,
        "top_p": 0.8,
        "temperature": 0.6,
    })
ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id

print("Loading TTS...")
tts = None
if not args.tts_model_id == "kokoro":
    raise NotImplemented("Kokoro is only supported at this time.")
else:
    tts = OpenAIText2SpeechTool(
            model_id=args.tts_model_id,
            voice="af_sky+af_bella", #single or multiple voicepack combo
            base_url="http://localhost:8880/v1",
            api_key="not-needed"
    )

def voice_record(tts):

    SECONDS_TO_RECORD = 5
    rate = 44100

    tts.stream_speech("Please ask a question.")

    recorder = sd.rec(
            SECONDS_TO_RECORD*rate,  
            samplerate=rate, 
            channels=2,
            dtype='float64'
    )

    print("\nAsk a question. Microphone is recording for ", 
          SECONDS_TO_RECORD, " seconds...\n")
    sd.wait()
    sf.write(args.audio_file, recorder, rate)

print("Initialization completed...")

while True:
    if args.demo_mode:
        voice_record(tts)

    start_time = time.time()
    docs = asr_loader.load()
    text = docs_loader.format_docs(docs)
    text = text.lstrip(' ').rstrip(' ')

    # RAG
    if args.rag:
        emb_res = embeddings.embed_query(text)
        vector_search_top_k = 1
        # Uncomment when RAG results are high quality...
        #score_threshold = 0.5
        #search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
        retriever = db.as_retriever(
            #search_type="similarity_score_threshold",
            search_kwargs={
                "k": vector_search_top_k,
                #"score_threshold": score_threshold
            }
        )

    # LLM templates (better results than without):
    # https://github.com/openvinotoolkit/openvino_notebooks/blob/21a745a6e3db36f58e37cf7f80d41a4816eb6c58/utils/llm_config.py#L57
    template = """<|im_start|>system\nProvide a short answer to the question based on the Provided Context below. If you do not know the answer say 'I do not know.'<|im_end|>\n<|im_start|>Provided Context\n{context}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"""

    prompt = PromptTemplate.from_template(template)

    print("\nQuestion asked: '" + text + "'\n")
    
    start_time_llm = time.time()
    generation_config = {
        "timeout": 120,
        "skip_prompt": True, 
        "pipeline_kwargs": 
        {"max_new_tokens": MAX_NEW_TOKENS}
    } 
    if args.rag:
        rag_chain = (
            {"context": retriever | docs_loader.format_docs, 
            "text": RunnablePassthrough()} 
            | prompt
            | ov_llm.bind(**generation_config)
            | StrOutputParser()
        )
    else:
        rag_chain = (
            {"context": lambda x: "", "text": RunnablePassthrough()}
            | prompt
            | ov_llm.bind(**generation_config)
            | StrOutputParser()
        )

    batch_size_for_tts = 10
    token_cnt = 0
    batch_tts = ""
    for token_val in rag_chain.stream(text):
        tts_text = token_val.strip()
        if tts_text == "":
            continue
        token_cnt = token_cnt + 1
        batch_tts = batch_tts + tts_text + " "
        # send "big enough" complete sentences
        if tts_text.endswith(".") and token_cnt >= batch_size_for_tts:
            batch_tts = batch_tts.rstrip()
            tts.stream_speech(batch_tts)
            batch_tts = ""
            token_cnt = 0
    batch_tts = batch_tts.rstrip()
    if not batch_tts == "":
        tts.stream_speech(batch_tts)
     
    if not args.demo_mode:
        break
    else:
        time.sleep(2)



