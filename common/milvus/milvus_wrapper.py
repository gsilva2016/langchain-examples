from typing import List
from pymilvus import (
    DataType,
    MilvusClient
)
from dotenv import load_dotenv
import os


class MilvusManager:
    def __init__(self, db_name="default", host="localhost", port="19530", env_file: str = ".env"):
        load_dotenv(env_file)

        milvus_dbname = os.getenv("MILVUS_DBNAME", db_name)
        milvus_host = os.getenv("MILVUS_HOST", host)
        milvus_port = os.getenv("MILVUS_PORT", port)
        
        self.milvus_client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}", db_name=milvus_dbname)
        print(f"Connected to Milvus at {milvus_host}:{milvus_port} using database {milvus_dbname}")
    
    def create_collection(self, collection_name: str, dim: int, overwrite=False):
        """
        Create a new collection in Milvus
        """
        try:
            if self.milvus_client.has_collection(collection_name):
                if overwrite:
                    print(f"Overwrite flag is set. Dropping existing collection {collection_name}.")
                    self.milvus_client.drop_collection(collection_name)
                else:
                    print(f"Collection {collection_name} already exists.")
                    self.milvus_client.load_collection(collection_name)
                    return

            schema = self.milvus_client.create_schema(enable_dynamic_field=True) 
            schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)

            index_params = MilvusClient.prepare_index_params()

            index_params.add_index(
                field_name="vector", 
                index_type="FLAT", 
                index_name="vector_index", 
                metric_type="COSINE", 
            )

            self.milvus_client.create_collection(
                collection_name=collection_name,
                index_params=index_params,
                schema=schema,
            )
        
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            raise e

        finally:
            res = self.milvus_client.get_load_state(collection_name=collection_name)
            print(f"Collection {collection_name} with dimension {dim}: Load state: {res}")
    
    def insert_data(self, collection_name: str, vectors: list, metadatas: list):
        """
        Insert data into the collection
        """
        try:
            
            if not self.milvus_client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist.")

            if metadatas is None:
                metadatas = [{} for _ in range(len(vectors))]

            records = []
            for vec, meta in zip(vectors, metadatas):
                records.append({
                    "vector": vec,
                    "metadata": meta
                })

            resp = self.milvus_client.insert(collection_name=collection_name, data=records)

            return {"status": "success", "total_chunks": resp["insert_count"]}

        except Exception as e:
            print(f"Error inserting data into {collection_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    def search(self, collection_name: str, query_vector: List[list] | list, limit: int = 1, filter: str = ""):
        """
        Search for similar vectors in the collection
        """
        try:
            if not self.milvus_client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist.")
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            results = self.milvus_client.search(
                collection_name=collection_name,
                data=query_vector,
                output_fields=["metadata", "vector"],
                limit=limit,
                filter=filter,
                search_params=search_params,
                consistency_level="Strong"
            )
            
            return list(results)

        except Exception as e:
            print(f"Error searching in {collection_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    def query(self, collection_name: str, filter: str = "", limit: int = 100):
        """
        Query the collection with an expression
        """
        try:            
            if not self.milvus_client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist.")
            
            results = self.milvus_client.query(
                collection_name=collection_name,
                filter=filter,
                output_fields=["pk", "metadata"],
                limit=limit
            )
            return results

        except Exception as e:
            print(f"Error querying {collection_name}: {e}")
            return {"status": "error", "message": str(e)}
