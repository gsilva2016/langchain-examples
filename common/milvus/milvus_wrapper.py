from datetime import datetime
from typing import List, Dict
import uuid
from pymilvus import Collection, connections, utility, db
from langchain_community.embeddings.openvino import OpenVINOEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document


class MilvusManager:
    def __init__(self, milvus_uri: str = "localhost",
                 milvus_port: int = 19530, 
                 milvus_dbname: str = "milvus_db",
                 txt_embedding_model: str = "sentence-transformers/all-mpnet-base-v2", 
                 txt_embedding_device: str = "GPU",
                 img_embedding_model: str = "openai/clip-vit-base-patch32",
                 img_embedding_device: str = "GPU"):
        """ 
        Initialize the MilvusManager class. Default values are set for the parameters if not provided.
        """
        self.milvus_uri = milvus_uri
        self.milvus_port = milvus_port
        self.milvus_dbname = milvus_dbname
        self.txt_embedding_model = txt_embedding_model
        self.txt_embedding_device = txt_embedding_device
        self.img_embedding_model = img_embedding_model
        self.img_embedding_device = img_embedding_device

        # Init the text embedding model - default is all-mpnet-base-v2 on GPU
        model_kwargs = {"device": self.txt_embedding_device}
        encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}
        self.ov_txt_embeddings = OpenVINOEmbeddings(
            model_name_or_path=self.txt_embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # Connect to Milvus
        self._connect_to_milvus()

        # Create an instance of the text vectorstore
        self.txt_vectorstore = Milvus(
            embedding_function=self.ov_txt_embeddings,
            collection_name="chunk_summaries",
            connection_args={"uri": f"http://{self.milvus_uri}:{self.milvus_port}", "db_name": self.milvus_dbname},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
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
            print(f"Collection {name} exists.")

    def embed_and_store(self, data: List[Dict]) -> Dict:
        """
        Embed text data and store it in Milvus.
        """
        try:
            documents = [
                Document(
                    page_content=item["chunk_summary"],
                    metadata={
                        "video_id": item["video_id"],
                        "chunk_id": item["chunk_id"],
                        "start_time": item["start_time"],
                        "end_time": item["end_time"],
                        "chunk_path": item["chunk_path"],
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                for item in data
            ]

            ids = [f"{doc.metadata['video_id']}_{doc.metadata['chunk_id']}_{uuid.uuid4()}" for doc in documents]

            self.txt_vectorstore.add_documents(documents=documents, ids=ids)

            return {"status": "success", "total_chunks": len(documents)}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def query(self, expr: str, collection_name: str = "chunk_summaries") -> Dict:
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
            results = self.txt_vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=None,
                include=["metadata"],
            )

            return {
                "status": "success",
                "results": [
                    {
                        "video_id": doc.metadata["video_id"],
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

    def get_txt_vectorstore(self) -> Milvus:
        """
        Get the text vector store instance.
        """
        return self.txt_vectorstore