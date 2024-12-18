import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import OpenVINOSpeechToTextLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import docs_loader_utils as docs_loader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ast

MAX_NEW_TOKENS = 8040

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--model_id", nargs="?", default="llmware/llama-3.2-3b-instruct-ov")
parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
parser.add_argument("--device", nargs="?", default="GPU")
args = parser.parse_args()

# Print Args
print("LangChain OpenVINO ASR+LLM - Video/Audio Summarization Demo")
print("This demonstrates LangChain using OpenVINO for ASR+LLM Video/Audio Summarization")
print("LLM model_id: ", args.model_id)
print("ASR model_id: ", args.asr_model_id)
print("Inference device  : ", args.device)
print("Audio file: ", args.audio_file)
#input("Press Enter to continue...")
# 

start_time = time.time()

# Edge local Speech-to-Text: https://huggingface.co/OpenVINO/distil-whisper-tiny-int4-ov
# Note: Requires LangChain patch
asr_loader = OpenVINOSpeechToTextLoader(args.audio_file, args.asr_model_id, device=args.device)
docs = asr_loader.load()
print("ASR docs created: ", len(docs))
#from langchain_core.load import dumpd, dumps, load, loads
#print("ARS results: ", docs)
#with open('LowerPrimary-Speakandwrite-transcript-asdoc.txt', 'w') as f:
#    f.write(str(docs))
#quit()
#with open('LowerPrimary-Speakandwrite-transcript-asdoc.txt', 'r') as f:
#    text = f.read()
#    print("text: ", text)
#    docs = load(text)
#    docs = ast.literal_eval(text)
#    print(docs)
#quit()
# Edge local LLM inference for text summarization
print("Starting LLM inference...")
model_cache_available = os.path.exists("./" + str(args.model_id))
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache-ov-models"}
ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": args.device, "ov_config": ov_config, "export": True if not model_cache_available else False}, # used for invoke
    pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True, "top_k": 10, "temperature": 1.0, "return_full_text": False, "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0, "use_cache": True},
)
ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id
if not os.path.exists("./" + str(args.model_id)):
    ov_llm.pipeline.model.save_pretrained("./" + str(args.model_id))
    ov_llm.pipeline.tokenizer.save_pretrained("./" + str(args.model_id))
else:
    print("Using OpenVINO model cache")

print("-------LLM Chapter Summary Generation---------")
# load ASR or text file for debugging
#text = docs_loader.format_docs(docs)
#with open('LowerPrimary-Speakandwrite-transcript.txt', 'w') as f:
#    f.write(text)
#with open('transcript.txt', 'r') as f:
#with open('mit_transcript.txt', 'r') as f:
#with open('LowerPrimary-Speakandwrite-transcript.txt', 'r') as f:
#    text = f.read()

#print("-----LLM text input-----")
#print(text)
#print("Number of chars: ", str(len(text)))

def get_summaries(transcript):

    for i in range(0, len(transcript)):
        doc_in = [transcript[i]]
        text = docs_loader.format_docs(doc_in)

        # sumllama
        instruction = "Please summarize the input document."
        row_json = [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"}]        
        formatted_prompt = ov_llm.pipeline.tokenizer.apply_chat_template(row_json, tokenize=False)
        summary = ov_llm.invoke(formatted_prompt)
        # remove LLM non-essential text
        # assistant\n\n
        summary = summary.replace('assistant\n\n', '')
        transcript[i].metadata["summary"] = summary


def get_chunked_docs(asr_docs, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap = 0,
                    length_function = len,
    )

    full_text = docs_loader.format_docs(asr_docs)
    doc_in = [Document(page_content=full_text,
        metadata={
            "language": 'en',
            "summary" : '',
            "topic": '',
            "start": '',
            "end": ''
        })]

    # chunk the docs by CHUNK_SIZE
    chunked_docs = text_splitter.split_documents(doc_in)
    #print(f"Generated {len(chunked_docs)} document chunks.")

    # fill chunked_docs with correct timestamps
    asr_ts = ast.literal_eval(asr_docs[0].metadata["timestamp"])
    chunk_idx = 0
    chunk_len = 0
    len_asr_docs = len(asr_docs)
    for i in range(0, len_asr_docs):
        chunk_len = chunk_len + len(asr_docs[i].page_content)
        if chunk_len > chunk_size:
            asr_ts_e = ast.literal_eval(asr_docs[i-1].metadata["timestamp"])
            #print("update: ", asr_ts, "= ", asr_ts[0], " ... " , asr_ts_e) # " ", chunk_idx, " orig: ", chunked_docs[chunk_idx].metadata["timestamp"])
            chunked_docs[chunk_idx].metadata["start"] = asr_ts[0]
            chunked_docs[chunk_idx].metadata["end"]   = asr_ts_e[0]
            #print("Updated chunked_doc: ", chunked_docs[chunk_idx])
            #print('\n\n')

            asr_ts = ast.literal_eval(asr_docs[i].metadata["timestamp"])
            chunk_idx = chunk_idx + 1
            chunk_len = chunk_len - chunk_size
#    print('\n\n')
#    print("Last chunk process: ", chunk_idx, " vs. ", len_asr_docs, " needed")
#    print(asr_docs[len_asr_docs-1].metadata["timestamp"])
#    print("Before: ", chunked_docs[chunk_idx])
#    print('\n\n')
    chunked_docs[chunk_idx].metadata["start"] = asr_ts_e[1]
    asr_ts = ast.literal_eval(asr_docs[len_asr_docs-1].metadata["timestamp"])
    chunked_docs[chunk_idx].metadata["end"] = "end" #asr_ts[1]
#    chunked_docs[chunk_idx].metadata["start"] = asr_ts[0]
#    print("After: ", chunked_docs[chunk_idx])

    return chunked_docs


docs = get_chunked_docs(docs, chunk_size=1000)
get_summaries(docs) #, chunk_size=1000) #61000) #5000)
print("Summary Results\n")
print(docs)
end_time = time.time()
print("\n\n")
print("LLM inference took: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")
