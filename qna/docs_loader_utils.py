import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
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
            if not file_path.endswith('.html') and not file_path.endswith('.pdf'):
                continue
            abs_path = os.path.join(TARGET_FOLDER, file_path)
            print(f"Loading document {abs_path} embedding into vector store...", flush=True)
            documents.extend(load_single_document(abs_path))

        spliter_name = "Character" #"RecursiveCharacter"  # PARAM
        chunk_size=1500  # PARAM
        chunk_overlap=0  # PARAM
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


def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)
