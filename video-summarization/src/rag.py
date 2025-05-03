import argparse
from common.milvus.milvus_wrapper import MilvusManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--query_text", type=str)
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="milvus_db")
    parser.add_argument("--collection_name", type=str, default="chunk_summaries")
    parser.add_argument("--retrive_top_k", type=int, default=1)
    args = parser.parse_args()
    
    milvus_manager = MilvusManager()
    vectorstore = milvus_manager.get_txt_vectorstore()
    if args.query_text:
        print(f"Search Query: {args.query_text}")
        docs = vectorstore.as_retriever(search_kwargs={"k": args.retrive_top_k}).invoke(args.query_text)
        
        if docs:
            print(f"Search Results: {docs}")
        else:
            print("No results found for the query.")
    
    else:
        print("No query text provided. Please provide a query text to search.")
       