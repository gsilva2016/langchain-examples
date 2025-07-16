import argparse
import subprocess
import os
from decord import VideoReader

from common.milvus.milvus_wrapper import MilvusManager

class RAG:
    def __init__(self, milvus_uri, milvus_port, milvus_dbname, collection_name):
        self.milvus_manager = MilvusManager(milvus_uri=milvus_uri, 
                                            milvus_port=milvus_port, 
                                            milvus_dbname=milvus_dbname, 
                                            collection_name=collection_name)
        self.vectorstore = self.milvus_manager.get_vectorstore()
        
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

            docs = self.vectorstore.similarity_search_with_score(query=query_text, k=retrive_top_k, expr=filter_expression if filter_expression else None)
                    
        elif query_img:
            print("Performing similarity search with image query:")
            print(f"Image Path: {query_img}")

            if filter_expression:
                print(f"With Filter Expression: {filter_expression}")
            
            emb = self.milvus_manager.ov_blip_embeddings.embed_image(query_img)

            docs = self.vectorstore.similarity_search_by_vector(embedding=emb, k=retrive_top_k, expr=filter_expression if filter_expression else None)
           
        elif filter_expression:
            print("Permforming query with filter expression (no similarity search since query_text is None):")

            docs = self.vectorstore.search_by_metadata(expr=filter_expression, limit=retrive_top_k)
            
        else:
            print("No query text and filter expression provided. Please set either of them in .env.")
        
        return docs
  
    def display_results(self, docs, save_video_clip=True, clip_length=5):
        if docs:
            for doc in docs:
                print(f"Similarity Score: {doc[1] if isinstance(doc, tuple) else 'Not Applicable'}")  
                doc = doc[0] if isinstance(doc, tuple) else doc
                for k, v in doc.metadata.items():
                    if k == "vector":
                        continue
                    print(f"{k}: {v}")
                print(f"{'-'*50}")
            print(f"Total documents retrieved: {len(docs)}")

            top_result = docs[0][0] if isinstance(docs[0], tuple) else docs[0]  
            chunk_path = top_result.metadata.get("chunk_path", None)
            frame_id = top_result.metadata.get("frame_id", None)

            if frame_id < 0:
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
    parser.add_argument("--milvus_dbname", type=str, default="milvus_db")
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