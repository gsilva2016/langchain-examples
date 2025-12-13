import os
from typing import List
from milvus_wrapper import MilvusManager
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name=os.getenv("NAME", "milvusserver"), host=os.getenv("MILVUS_HOST", "localhost"), port=int(os.getenv("PORT", 8001)), json_response=True)
milvus = MilvusManager()
DEBUG = os.getenv("DEBUG", "FALSE").upper() == "TRUE"

@mcp.tool()
def list_collections():
    """
    List all collections in the Milvus database
    """
    print("Calling list_collections")
    client = milvus._get_client()
    result = client.list_collections()
    
    if DEBUG:
        print(f"List Collections result: {result}")
        
    return result

@mcp.tool()
def create_collection(collection_name: str, dimension: int, overwrite: bool = False):
    """
    Create a new collection in Milvus
    """
    print(f"Calling create_collection with parameters: collection_name={collection_name}, dimension={dimension}, overwrite={overwrite}")
    result = milvus.create_collection(collection_name=collection_name, dim=dimension, overwrite=overwrite)
    
    if DEBUG:
        print(f"Create Collection result: {result}")
        
    return result

@mcp.tool()
def insert_data(collection_name: str, vectors: list, metadatas: list, partition_name: str = None):
    """
    Insert a batch of vectors into a collection
    """
    print(f"Calling insert_data with parameters: collection_name={collection_name}, number_of_vectors={len(vectors)}, partition_name={partition_name}")
    result = milvus.insert_data(collection_name=collection_name, vectors=vectors, 
                                   metadatas=metadatas, partition_name=partition_name)
    
    if DEBUG:
        print(f"Insert Data result: {result}")
        
    return result

@mcp.tool()
def search(collection_name: str, query_vector: List[list] | list, output_fields: list = ["metadata", "vector"],
               limit: int = 1, filter: str = "", partition_names: list = None):
    """
    Search for similar vectors in a collection
    """
    print(f"Calling search with parameters: collection_name={collection_name}, number_of_query_vectors={len(query_vector) if isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list) else 1}, output_fields={output_fields}, limit={limit}, filter={filter}, partition_names={partition_names}")
    result = milvus.search(collection_name=collection_name, query_vectors=query_vector, 
                                   output_fields=output_fields,
                                   top_k=limit, partition_name=partition_names)
    if DEBUG:
        print(f"Search result: {result}")
        print(f"Search result: length={len(result)}")
        
    return result

@mcp.tool()
def upsert_data(collection_name: str, pks: list, vectors: list, metadatas: list, partition_name: str = None):
    """
    Upsert data into the collection
    """
    print(f"Calling upsert_data with parameters: collection_name={collection_name}, number_of_vectors={len(vectors)}, partition_name={partition_name}")
    result = milvus.upsert_data(collection_name=collection_name, pks=pks, vectors=vectors, metadatas=metadatas, partition_name=partition_name)
    
    if DEBUG:
        print(f"Upsert Data result: {result}")
        
    return result

@mcp.tool()
def query(collection_name: str, filter: str = "", limit: int = 100, output_fields: list = ["pk", "metadata"], partition_names: list = None):
    """
    Query the collection with an expression
    """
    print(f"Calling query with parameters: collection_name={collection_name}, filter={filter}, limit={limit}, output_fields={output_fields}, partition_names={partition_names}")
    result = milvus.query(collection_name=collection_name, filter=filter, limit=limit, output_fields=output_fields, partition_names=partition_names)
    
    if DEBUG:
        print(f"Query result: {result}")
        print(f"Query result: length={len(result)}")
        
    return result

@mcp.tool()
def drop_collection(collection_name: str):
    """
    Drop a collection in Milvus
    """
    print(f"Calling drop_collection with parameters: collection_name={collection_name}")
    result = milvus.drop_collection(collection_name=collection_name)
    
    if DEBUG:
        print(f"Drop Collection result: {result}")
        
    return result

if __name__ == "__main__":
    mcp.run(transport=os.getenv("TRANSPORT", "streamable-http"))