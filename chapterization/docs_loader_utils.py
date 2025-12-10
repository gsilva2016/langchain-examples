import os
import sys
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from datetime import timedelta
import json
#from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, )

# RAG Docs
TARGET_FOLDER = "./docs/"

TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
}

LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
def load_docs(embeddings, skip_docs_loading):
    if not skip_docs_loading:
        documents = []    
        for file_path in os.listdir(TARGET_FOLDER):
            if not file_path.endswith('.html'):
                continue
            abs_path = os.path.join(TARGET_FOLDER, file_path)
            print(f"Loading document {abs_path} embedding into vector store...", flush=True)
            documents.extend(load_single_document(abs_path))

        spliter_name = "RecursiveCharacter"  # PARAM
        chunk_size=1000  # PARAM
        chunk_overlap=200  # PARAM
        text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embeddings)  # This command populates vector store with embeddings
    else:
        dimensions: int = len(embeddings.embed_query("get_dims"))
        db = FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(dimensions),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            )
    return db


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")

def pretty_print_docs(docs):
    buff = (
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

    print(str(buff)[:500])


def chunk_transcript_docs(docs, chunk_size = 1500):
    if chunk_size == 0:
        chunk_size = sys.maxsize
    new_docs = []
    text_len_batched = 0
    text_ts_start = None
    text_ts_end = None
    text_batched = ""
    num_docs = len(docs) - 1

    for i, doc in enumerate(docs):
        text = doc.page_content
        text_len_batched += len(text)
        text_batched += text

        if text_ts_start is None:
            text_ts_start = doc.metadata["timestamp"]
            
        if (text_len_batched >= chunk_size) or (text_len_batched >0 and i == num_docs):
            text_ts_end = eval(doc.metadata["timestamp"])

            # Set an end time if it is missing from transcript
            if text_ts_end[1] is None:
                text_ts_end = (text_ts_end[0], text_ts_end[0] + 1)

            new_doc = Document(
                        page_content=text_batched.lstrip(),
                        metadata={
                            "start": str(eval(text_ts_start)[0]),
                            "end": str(text_ts_end[1])
                        })
            new_docs.append(new_doc)
            text_ts_start = None
            text_ts_end = None
            text_batched = ""
            text_len_batched = 0

    return new_docs


def format_seconds_to_hh_mm_ss_ms(total_seconds_float):
    # Create a timedelta object from the total seconds
    td = timedelta(seconds=total_seconds_float)

    # Get the total seconds from the timedelta object (integer part)
    total_seconds_int = int(td.total_seconds())

    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(total_seconds_int, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Calculate milliseconds from the fractional part of the original float
    milliseconds = int((total_seconds_float - total_seconds_int) * 1000)

    # Format the output string
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def format_transcript_docs_sectionize_big(docs_big, docs):
    ret_str = "[ "

    for i in range(0, len(docs_big)):
        doc = docs_big[i]
        master_end_time = eval(doc.metadata['end'])
        master_start_time = eval(doc.metadata['start'])
        for j in range(0, len(docs)):
            doc_details = docs[j]
            start = eval(doc_details.metadata['start'])
            end = eval(doc_details.metadata['end'])
            doc_text = doc_details.page_content

            if not master_end_time is None and end is None:
                end = master_end_time + 1
            elif master_end_time is None and end is None:
                print("SHOULD NOT HAPPEN")
                quit()
            elif master_end_time is None:
                print("SHOULD NOT HAPPEN2")
                quit()
            if start < master_start_time:
                continue
            elif end > master_end_time:
                break

            ret_str = ret_str + " { \"start\": \"" + str(start) + "\", \"end\": \"" + str(end) + "\", \"text\": \"" + doc_text + "\" },"

    ret_str = ret_str[:-1]
    return ret_str + " ]"

def format_summary(summary):
    lines = summary.splitlines()
    new_summary = ""

    for line in lines:
        if "start: \"" in line or "end: \"" in line:
            newline = line.split('"')
            formatted_ts = "\"" + format_seconds_to_hh_mm_ss_ms(float(newline[1])) + "\""
            newline[1] = formatted_ts
            new_summary = new_summary + " ".join(newline) + "\n"
        else:
            new_summary = new_summary + line + "\n"

    return new_summary

def format_transcript_docs(docs):
    return " ".join("start: \"" + doc.metadata['start'] + "\", end: " + doc.metadata['end'] + "; " + doc.page_content + '\n' for doc in docs)

def format_docs_ts(docs):
    for i in range(0, len(docs)):
        doc = docs[i]
        lst = json.loads(doc.page_content)

        for j in range(0, len(lst)):
            lst[j]['start'] = format_seconds_to_hh_mm_ss_ms(float(lst[j]['start']))
            lst[j]['end'] = format_seconds_to_hh_mm_ss_ms(float(lst[j]['end']))

        docs[i].page_content = json.dumps(lst)
    return docs

def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)
