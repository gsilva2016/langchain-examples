import argparse
from time import sleep

from common.milvus.milvus_wrapper import MilvusManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--query_text", type=str)
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="milvus_db")
    parser.add_argument("--collection_name", type=str, default="video_chunks")
    parser.add_argument("--retrive_top_k", type=int, default=5)
    parser.add_argument("--filter_expression", type=str, nargs="?")
    args = parser.parse_args()

    
    milvus_manager = MilvusManager()
    vectorstore = milvus_manager.get_vectorstore()
    
    if args.query_text:
        print(f"Search Query has been provided: {args.query_text}")
        
        if args.filter_expression:
            print(f"With Filter Expression: {args.filter_expression}")
        
        docs = vectorstore.similarity_search_with_score(query=args.query_text, k=args.retrive_top_k, expr=args.filter_expression if args.filter_expression else None)
        if docs:
            for doc, score in docs:
                print(f"Document: {doc.page_content}\nScore: {score}\nMetadata: {doc.metadata}\n{'-'*50}")
            print(f"Total documents retrieved: {len(docs)}")
        else:
            print("No results found for the provided query.")
    else:
        print("No query text provided. Please provide a query text to search.")
       