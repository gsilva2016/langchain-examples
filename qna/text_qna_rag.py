import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
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
import readline


MAX_NEW_TOKENS = 200

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", nargs="?", default="llmware/qwen2-0.5b-chat-ov")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--llm_batch_size", default=1, type=int)
parser.add_argument("--demo_mode", default=False, action="store_true")
parser.add_argument("--rag", default=False, action="store_true")
args = parser.parse_args()

print("LangChain OpenVINO Text QnA")
print("LLM model_id: ", args.model_id)
print("LLM batch_size: ", args.llm_batch_size)
print("Inference device  : ", args.device)
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
        "temperature": 0.1,
    })
ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id


print("Initialization completed...")

while True:
    text = input("Enter your question: ") 

    start_time = time.time()
    text = text.lstrip(' ').rstrip(' ')
    print("\nQuestion asked: '" + text + "'\n")

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

    for token_val in rag_chain.stream(text):
        print(token_val, end="", flush=True)

    print("\n\n")
     
    if not args.demo_mode:
        break
    else:
        time.sleep(2)



