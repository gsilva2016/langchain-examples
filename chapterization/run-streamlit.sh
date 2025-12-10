source .env

source activate-conda.sh
activate_conda
conda activate $CONDA_STREAMLIT_ENV_NAME

streamlit run streamlit-demo.py --server.port=8080
