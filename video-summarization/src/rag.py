import argparse
from time import sleep

from langchain_openvino_multimodal.embeddings import OpenVINOClipEmbeddings
from common.milvus.milvus_wrapper import MilvusManager

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--query_text", type=str)
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="milvus_db")
    parser.add_argument("--collection_name", type=str, default="chunk_summaries")
    parser.add_argument("--retrive_top_k", type=int, default=2)
    parser.add_argument("--filter_expression", type=str, nargs="?")
    args = parser.parse_args()

    
    milvus_manager = MilvusManager()
    txt_vectorstore = milvus_manager.get_txt_vectorstore()
    img_vectorstore = milvus_manager.get_img_vectorstore()
    
    if args.query_text:
        print(f"Search Query: {args.query_text}")
            
        if args.filter_expression:
            docs = img_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.retrive_top_k, "expr": args.filter_expression}
            ).invoke(args.query_text)
        else:
            docs = img_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.retrive_top_k}
            ).invoke(args.query_text)
                    
        if docs:
            print(f"Retrieved {len(docs)} results.")
            pretty_print_docs(docs)
        else:
            print("No results found for the query.")
    
    else:
        print("No query text provided. Please provide a query text to search.")
       