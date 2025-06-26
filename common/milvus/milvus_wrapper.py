from datetime import datetime
from typing import List, Dict
import uuid
import os
from pymilvus import Collection, connections, utility, db
from langchain_milvus import Milvus
from langchain_openvino_multimodal import OpenVINOBlipEmbeddings
from dotenv import load_dotenv


class MilvusManager:
    def __init__(self, milvus_uri: str = "localhost",
                 milvus_port: int = 19530, 
                 milvus_dbname: str = "milvus_db",
                 env_file: str = ".env",
                 embedding_model: str = "Salesforce/blip-itm-base-coco",
                 txt_embedding_device: str = "GPU",
                 img_embedding_device: str = "GPU",
                 collection_name: str = "video_chunks") -> None:
        """ 
        Initialize the MilvusManager class. Default values are set for the parameters if not provided.
        """
        load_dotenv(env_file)

        self.milvus_uri = milvus_uri
        self.milvus_port = milvus_port
        self.milvus_dbname = milvus_dbname
        self.embedding_model = os.getenv("EMBEDDING_MODEL", embedding_model)
        self.txt_embedding_device = os.getenv("TXT_EMBEDDING_DEVICE", txt_embedding_device)
        self.img_embedding_device = os.getenv("IMG_EMBEDDING_DEVICE", img_embedding_device)
        
        self.ov_blip_embeddings = OpenVINOBlipEmbeddings(model_id=self.embedding_model, ov_text_device=self.txt_embedding_device,
                                                        ov_vision_device=self.img_embedding_device)

        # Connect to Milvus
        self._connect_to_milvus()

        self.vectorstore = Milvus(
            embedding_function=self.ov_blip_embeddings,
            collection_name=collection_name,
            connection_args={"uri": f"http://{self.milvus_uri}:{self.milvus_port}", "db_name": self.milvus_dbname},
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            consistency_level="Strong",
            drop_old=False,
        )

    def _connect_to_milvus(self):
        """
        Connect to the Milvus database and set up the database.
        """
        connections.connect(host=self.milvus_uri, port=self.milvus_port)
        if self.milvus_dbname not in db.list_database():
            db.create_database(self.milvus_dbname)
        db.using_database(self.milvus_dbname)

        collections = utility.list_collections()
        for name in collections:
            # Not droppingthe collection if it exists for now
            # utility.drop_collection(name)
            print(f"Collection {name} exists.")

    def embed_txt_and_store(self, data: List[Dict]) -> Dict:
        """
        Embed text data and store it in Milvus.
        """
        try:
            if not data:
                return {"status": "error", "message": "No data to embed", "total_chunks": 0}
            
            all_summaries = [item["chunk_summary"] for item in data]
            embeddings = self.ov_blip_embeddings.embed_documents(all_summaries)
            print(f"Generated {len(embeddings)} text embeddings of Shape: {embeddings[0].shape}")
            
            # Prepare texts and metadata
            texts = [item["chunk_summary"] for item in data]
            metadatas = [
                {
                    "video_path": item["video_path"],
                    "chunk_id": item["chunk_id"],
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "chunk_path": item["chunk_path"],
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "frame_id": -1, # not required for text but required field for metadata since image needs it
                    "mode": "text",
                }
                for item in data
            ]

            ids = [f"{meta['chunk_id']}_{uuid.uuid4()}" for meta in metadatas]
            self.vectorstore.add_embeddings(texts=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)

            return {"status": "success", "total_chunks": len(texts)}

        except Exception as e:
            print(f"Error in embedding and storing text data: {e}")
            return {"status": "error", "message": str(e), "total_chunks": 0}
    
    def embed_img_and_store(self, chunk: Dict) -> Dict:
        """
        Embed image data and store it in Milvus.
        """
        try:
            all_sampled_images = chunk["frames"]
            embeddings = self.ov_blip_embeddings.embed_images(all_sampled_images)
            print(f"Generated {len(embeddings)} img embeddings of Shape: {embeddings[0].shape}")
            
            # Prepare texts and metadata
            texts = [chunk["chunk_path"] for emb in embeddings]
            metadatas = [
                {
                    "video_path": chunk["video_path"],
                    "chunk_id": chunk["chunk_id"],
                    "frame_id": idx,
                    "start_time": float(chunk["start_time"]),
                    "end_time": float(chunk["start_time"]),
                    "chunk_path": chunk["chunk_path"],
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "mode": "image",
                }
                for idx in chunk["frame_ids"]
            ]

            ids = [f"{meta['chunk_id']}_{uuid.uuid4()}" for meta in metadatas]
            self.vectorstore.add_embeddings(texts=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)

            return {"status": "success", "total_frames_in_chunk": len(all_sampled_images)}

        except Exception as e:
            print(f"Error in embedding and storing images: {e}")
            return {"status": "error", "message": str(e), "total_frames_in_chunk": 0}

    def query(self, expr: str, collection_name: str = "video_chunks") -> Dict:
        """
        Query data from Milvus using an expression.
        """
        try:
            collection = Collection(collection_name)
            collection.load()

            results = collection.query(expr, output_fields=["chunk_id", "chunk_path", "video_id"])
            print(f"{len(results)} vectors returned for query: {expr}")

            return {"status": "success", "chunks": results}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search(self, query: str, top_k: int = 1) -> Dict:
        """
        Perform similarity search in Milvus.
        """
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=None,
                include=["metadata"],
            )

            return {
                "status": "success",
                "results": [
                    {
                        "video_path": doc.metadata["video_path"],
                        "chunk_id": doc.metadata["chunk_id"],
                        "start_time": doc.metadata["start_time"],
                        "end_time": doc.metadata["end_time"],
                        "chunk_path": doc.metadata["chunk_path"],
                        "chunk_summary": doc.page_content,
                    }
                    for doc in results
                ],
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_vectorstore(self) -> Milvus:
        """
        Get the vector store instance.
        """
        return self.vectorstore    