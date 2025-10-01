import argparse
import subprocess
import os
from decord import VideoReader

from dotenv import load_dotenv
from common.milvus.milvus_wrapper import MilvusManager
from langchain_openvino_multimodal import OpenVINOBlipEmbeddings


class RAG:
    def __init__(self, milvus_uri: str, milvus_port: str, milvus_dbname: str, collection_name: str):
        self.milvus_manager = MilvusManager(host=milvus_uri, 
                                            port=milvus_port, 
                                            db_name=milvus_dbname)
        load_dotenv()
        embedding_model = os.getenv("EMBEDDING_MODEL", "Salesforce/blip-itm-base-coco")
        txt_embedding_device = os.getenv("TXT_EMBEDDING_DEVICE", "GPU")
        img_embedding_device = os.getenv("IMG_EMBEDDING_DEVICE", "GPU")
        self.ov_blip_embedder = OpenVINOBlipEmbeddings(model_id=embedding_model, ov_text_device=txt_embedding_device,
                                                        ov_vision_device=img_embedding_device)
        self.test = "true"
        
        self.collection_name = collection_name
        
        if self.milvus_manager.milvus_client.has_collection(collection_name):
            print(f"Collection {collection_name} already exists.")
            self.milvus_manager.milvus_client.load_collection(collection_name)
        else:
            raise ValueError(f"Collection {collection_name} does not exist.")

    def _extract_clip(self, frame_id, chunk_path, clip_length=5):
        print(f"\nSaving {clip_length}-second clip for top result")
        
        # using decord here since package is already installed via pip and is used for sampling frames module
        vr = VideoReader(chunk_path)
        fps = vr.get_avg_fps()
        if not fps or fps == 0:
            raise ValueError(f"Unable to find FPS for this video: {chunk_path}")

        frame_time = frame_id / fps
        start_time = max(frame_time - clip_length / 2, 0)

        os.makedirs("rag_clips", exist_ok=True)
        output_path = f"rag_clips/clip_frame_{frame_id}.mp4"
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", chunk_path,
            "-c:v", "libx264",
            output_path,
            "-y"
        ]
        subprocess.run(cmd, check=True)
        print(f"Clip saved successfully at {output_path} from chunk video: {chunk_path}")

    def run(self, query_text=None, query_img=None, retrive_top_k=5, filter_expression=None):
        docs = None
        
        if query_text:
            print("Performing similarity search with the following parameters:")
            print(f"Search Query: {query_text}")

            if filter_expression:
                print(f"With Filter Expression: {filter_expression}")

            embedding = self.ov_blip_embedder.embed_query(query_text)
            print(f"Generated text embedding of Shape: {embedding.shape}")
            
            docs = self.milvus_manager.search(collection_name=self.collection_name,
                                            query_vector=[embedding],
                                            limit=retrive_top_k,
                                            filter=filter_expression if filter_expression else "")
            docs = [item for sublist in docs for item in sublist]

                    
        elif query_img:
            print("Performing similarity search with image query:")
            print(f"Image Path: {query_img}")

            embedding = self.ov_blip_embedder.embed_image(query_img)
            print(f"Generated img embedding of Shape: {embedding.shape}")
            
            docs = self.milvus_manager.search(collection_name=self.collection_name,
                                            query_vector=[embedding],
                                            limit=retrive_top_k,
                                            filter=filter_expression if filter_expression else "")
            docs = [item for sublist in docs for item in sublist]
           
        elif filter_expression:
            print("Permforming query with filter expression (no similarity search since query_text is None):")

            docs = self.milvus_manager.query(collection_name=self.collection_name, 
                                            filter=filter_expression if filter_expression else "", 
                                            limit=retrive_top_k)
        else:
            print("No query text and filter expression provided. Please set either of them in .env.")
        
        return docs
  
    def display_results(self, docs, save_video_clip=True, clip_length=5):
        if docs:
            for doc in docs:
                print(f"Similarity Score: {doc['distance'] if 'distance' in doc else 'Not Applicable'}")
                if "entity" in doc:
                    print(f"All Metadata: {doc['entity']['metadata'] if 'metadata' in doc['entity'] else 'No metadata available'}")
                elif "metadata" in doc:
                    print(f"All Metadata: {doc['metadata']}")
                else:
                    print("No metadata available for this document.")
                print(f"{'-'*50}")
                
            print(f"Total documents retrieved: {len(docs)}")

            top_result = docs[0]
            if "entity" in top_result:
                chunk_path = top_result["entity"]["metadata"].get("chunk_path", None)
                frame_id = top_result["entity"]["metadata"].get("frame_id", None)
            else:
                chunk_path = top_result["metadata"].get("chunk_path", None)
                frame_id = top_result["metadata"].get("frame_id", None)

            if not frame_id:
                print("Search retrieved result based on the chunk summary (text). You may find the chunk video associated with this summary at the following path: ", chunk_path)
            else:
                if save_video_clip:
                    self._extract_clip(frame_id, chunk_path, clip_length)

        else:
            print("No results found for the provided query/filter expression.")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--query_text", type=str, default=None)
    parser.add_argument("--query_img", type=str, default=None)
    parser.add_argument("--milvus_uri", type=str, default="localhost")
    parser.add_argument("--milvus_port", type=int, default=19530)
    parser.add_argument("--milvus_dbname", type=str, default="default")
    parser.add_argument("--collection_name", type=str, default="video_chunks")
    parser.add_argument("--retrieve_top_k", type=int, default=5)
    parser.add_argument("--filter_expression", type=str, nargs="?")
    parser.add_argument("--save_video_clip", type=bool, default=True)
    parser.add_argument("--video_clip_duration", type=int, default=5)

    args = parser.parse_args()
    
    rag = RAG(milvus_uri=args.milvus_uri, 
              milvus_port=args.milvus_port, 
              milvus_dbname=args.milvus_dbname,
              collection_name=args.collection_name)

    docs = rag.run(query_text=args.query_text,
            query_img=args.query_img,
            retrive_top_k=args.retrieve_top_k,
            filter_expression=args.filter_expression)
    
    rag.display_results(docs, args.save_video_clip, args.video_clip_duration)
