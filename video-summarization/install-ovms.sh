export PATH=$PATH:${PWD}/ovms/bin
if command -v ovms &> /dev/null; then
    echo "OpenVINO Model Server (OVMS) is already installed."
    return 0
fi

echo "Install OpenVINO Model Server (OVMS) on baremetal"	
source activate-conda.sh
activate_conda
conda create -n ovms_env python=3.12 -y
conda activate ovms_env
conda install pip -y

# Install dependencies
sudo apt update -y && sudo apt install -y libxml2 curl
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/requirements.txt -o ovms_requirements.txt
pip install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
pip install -r ovms_requirements.txt

# Download OVMS
wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu24_python_on.tar.gz
tar -xzvf ovms_ubuntu24_python_on.tar.gz

if [ "$1" == "--skip" ]; then
    echo "Skipping OpenVINO optimized model file creation"
else
    echo "Creating OpenVINO optimized model files for MiniCPM"
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models
    
    output=$(python export_model.py text_generation --source_model openbmb/MiniCPM-V-2_6 --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2 --pipeline_type VLM 2>&1 | tee /dev/tty)

    if echo "$output" | grep -q "Tokenizer won't be converted."; then
        echo ""
        echo "Error: Tokenizer was not converted successfully, OVMS export model has partially errored. Please check the logs."
        exit 1
    fi
fi