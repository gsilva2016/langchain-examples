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

This demo is in a Jupyter Notebook format. Please go to `vision_rag_image_search.ipynb` and follow the instructions annotated as part of the Notebook. 

## Pre-Requisites

1. Verified on Linux - Ubuntu 24.04 setup - Intel CPU and iGPU setup
2. Python >= 3.10
3. Jupyter Environment
4. `product_list.pdf` - This PDF should include all the food items and their corresponsing product codes in a PDF format

## Installation

You may choose to run the Notebook with Jupyter or Jupyter Lab. We also recommend running the Notebook in a virtual environment or Conda env to ensure clean package install.

1. First install [Conda](https://github.com/conda-forge/miniforge) or Python Virtual Environment

2. If using Python Virtual Environment: 

    ```
    python -m venv myenv
    source myenv/bin/activate 
    ```

3. If using Conda:

    ```
    conda create -n myenv python=3.10
    conda activate
    ```

4. Install either Jupyter Lab or Classic Jupyter Notebook. 
    - Jupyter Lab: `pip install jupyterlab`. Once installed, launch JupyterLab with: `jupyter lab`
    - Install the classic Jupyter Notebook with: `pip install notebook`. To run the notebook: `jupyter notebook`

5. Add virtual environment to Jupyter kernel:

    ```
    pip install --user ipykernel
    python -m ipykernel install --user --name=myenv
    ```

## Run Notebook

```
cd vision_rag/
jupyter lab vision_rag_image_search.ipynb
```
You may begin interacting with the Notebook using the instructons provided in the Notebook. 

P.S: Ensure the kernel listed on the top right is your custom virtual environment created above (myenv for example)



