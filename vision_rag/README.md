## Vision RAG with FastSAM, CLIP, FAISS and MiniCPM

This project demonstrates identification of food items from a food tray using open-source zero shot object detection models without finetuning, open-source embeddings models without finetuning and FAISS as the vector database. All models run locally on an Intel Core platform.

Core pipeline consists of:
1. Metadata and Image extraction from menu PDF document
2. FastSAM for zero shot object detection
3. Custom filter function to reduce false positives in FastSAM
4. OpenVINO for optimized model inference on Intel platforms
5. CLIP model for main embeddings
6. Custom Augmentation function to increase embeddings due to less data
7. Image identification using CLIP and FAISS (open source models without finetuning)
8. Synthesis of vector DB retrieved data using an LVM (MiniCPM)

This demo is in a Jupyter Notebook format. Please go to `vision_rag_image_search.ipyng` and follow the instructions annotated as part of the Notebook. 

