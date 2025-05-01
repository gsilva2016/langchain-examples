import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from api_requests import search_in_milvus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--query_text", type=str)
    args = parser.parse_args()
    
    with ThreadPoolExecutor() as pool:
        if args.query_text:
            print(f"Searh Query: {args.query_text}")
            query_text = args.query_text
            query_future = pool.submit(search_in_milvus, query_text)
    
            if query_future and query_future.result():
                print(f"Search Results: {ast.literal_eval(query_future.result().decode('utf-8'))}")
        
        else:
            print("No query text provided. Please provide a query text to search.")