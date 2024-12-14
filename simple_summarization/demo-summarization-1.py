import argparse
import time
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
input("Press Enter to continue...")
# 

# Edge local Speech-to-Text: https://huggingface.co/OpenVINO/distil-whisper-tiny-int4-ov
# Note: Requires LangChain patch
asr_loader = OpenVINOSpeechToTextLoader(args.audio_file, args.asr_model_id, device=args.device)
docs = asr_loader.load()
#print("ARS results: ", docs)


# Edge local LLM inference for text summarization
print("Starting LLM inference...")
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
#"num_return_sequences"
# temp: 0.2
#  "encoder_repetition_penalty": 1.5
ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": "CPU", "ov_config": ov_config}, # used for invoke
    pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True, "top_k": 10, "temperature": 1.0, "return_full_text": True, "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0, "use_cache": True},
)

ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id

template = """Your task is to improve the user input's readability by adding punctuation if needed, removing verbal tics, and structure the text in paragraphs separated with '\n\n' for the following: {text}.
Keep the wording as faithful as possible to the original text. 
Put your answer within <answer></answer> tags."""

template = """system: Identify all the relevant sections/topics in the following text: {context}
For each section identified  Create a Section Title and Create a Section Summary and keep the  wording as faithful as possible to the original text"""

template = """
You are a helpful assistant.

Keep the wording as faithful as possible to the original text. Create all relevant sections, section titles, and paraphrased section summaries for the following text: 

{context}"""

template = """
Create concise sections of the main topics. Create all relevant sections, section titles, and paraphrased section summaries. Keep the wording as faithful as possible to the original text provided by the user/human.

Human: Generate concise sections for the main topics in: {context}"""

#template = """
#Create consise sections of the main topics from the text provided by the user/human. For each section you create give title as well. Keep the wording as faithful as possible to the original text provided by the user/human. Do not add any information that is not provided by the human/user.
#"""

#template = """You are a helpful assistant.

#Your task is to structure a given text into paragraphs. Keep the wording as close as possible to the original text. 

#Format your result as a list of paragraphs.
#Example:
#1. I am a paragraph.
#2. I am another paragraph. So on, and so forth. 
#Keep creating paragraphs like this.
#3. This is yet another paragraph!

#User: Format the user's input by structuring the provided text into paragraphs for the following: 

#{context}

#"""

prompt = ChatPromptTemplate.from_messages([("system", template)])
chain = create_stuff_documents_chain(ov_llm, prompt)

print("-------LLM Chapter Summary Generation---------")
start_time = time.time()

# load ASR or text file for debugging
text = docs_loader.format_docs(docs)

#with open('transcript.txt', 'r') as f:
#with open('mit_transcript.txt', 'r') as f:
#    text = f.read()

#print("-----LLM text input-----")
#print(text)
#print("Number of chars: ", str(len(text)))

def transcript_to_paragraphs(transcript_as_text, chain, chunk_size=5000):

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =chunk_size,
            chunk_overlap  = 0,
            length_function = len,
            )

    doc_in = [Document(
            page_content=transcript_as_text,
            metadata={
                "language": 'en'
            })]
    
    split_docs = text_splitter.split_documents(doc_in)    
    print(f"Generated {len(split_docs)} documents.")
    for i in range(0, len(split_docs)):
        doc_in = [split_docs[i]]
        print("Input text:", doc_in)
        paragraph_created = chain.invoke({"context": doc_in})
        print("Paragraph/Section ", str(i), " created: \n", paragraph_created)


transcript_to_paragraphs(text, chain, chunk_size=1000) #61000) #5000)
end_time = time.time()
print("\n\n")
print("LLM inference took: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")
