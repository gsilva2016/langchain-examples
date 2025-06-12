#!/bin/bash

HUGGINGFACE_TOKEN=

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN variable"
    exit 1
fi

dpkg -s sudo &> /dev/null
if [ $? != 0 ]
then
	DEBIAN_FRONTEND=noninteractive apt update
	DEBIAN_FRONTEND=noninteractive apt install sudo -y
fi

# Set target device for model export
DEVICE="GPU"

# Install Conda
source activate-conda.sh

# One-time installs
if [ "$1" == "--skip" ]; then
	echo "Skipping dependencies"
	activate_conda
else    
    echo "Installing dependencies"
	sudo DEBIAN_FRONTEND=noninteractive apt update
	sudo DEBIAN_FRONTEND=noninteractive apt install git ffmpeg wget -y

	CUR_DIR=`pwd`
    cd /tmp
    miniforge_script=Miniforge3-$(uname)-$(uname -m).sh
    [ -e $miniforge_script ] && rm $miniforge_script
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/$miniforge_script"
    bash $miniforge_script -b -u
    # used to activate conda install
    activate_conda
    conda init
    cd $CUR_DIR

    # neo/opencl drivers 24.45.31740.9
    mkdir neo
    cd neo
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb
    sudo dpkg -i *.deb
    # sudo apt install ocl-icd-libopencl1
    cd ..
	
fi

echo "Installing Milvus as a standalone service"
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing Docker"

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    # Check if the key was added successfully
    if [ $? -ne 0 ]; then
        echo "Failed to add Docker GPG key. Please check your network connection or the URL."
        exit 1
    fi
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    sudo docker run hello-world
    # check if last command was successful
    if [ $? -ne 0 ]; then
        echo "Docker installation failed. Please check the installation logs."
        exit 1
    fi
    echo "Docker installed successfully."

    # Add user to the docker group to prevent permission issues
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker

fi

echo "Proceeding with Milvus setup"
echo "Downloading and running Milvus"
echo ""
if [ ! -e standalone_embed.sh ]; then
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
fi

# Check if Milvus is already running
if docker ps | grep -q milvus; then
    echo "Milvus is already running."
    echo ""

else
    echo "Starting Milvus..."
    bash standalone_embed.sh start
    echo "Milvus has been started. It is running at http://localhost:19530"
    echo ""
fi

echo "You can check the status of Milvus using the following command:"
echo "docker ps | grep milvus"
echo ""

echo "You can stop Milvus using the following command:"
echo "bash standalone_embed.sh stop"
echo ""

echo "You can delete Milvus data using the following command:"
echo "bash standalone_embed.sh delete"
echo ""

# Install OpenVINO Model Server (OVMS) on baremetal
export LD_LIBRARY_PATH=${PWD}/ovms/lib:$LD_LIBRARY_PATH
export PATH=$PATH:${PWD}/ovms/bin
export PYTHONPATH=${PWD}/ovms/lib/python:$PYTHONPATH
if command -v ovms &> /dev/null; then
    echo "OpenVINO Model Server (OVMS) is already installed."
else
    echo "Installing OpenVINO Model Server (OVMS) on baremetal"
    # Download OVMS .deb package
    wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu24_python_on.tar.gz
    tar -xzvf ovms_ubuntu24_python_on.tar.gz
    sudo apt update -y && sudo apt install -y libxml2 curl
    sudo apt -y install libpython3.12
    pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
fi
    
# Create python environment
# if conda environment already exists, skip creation
if conda env list | grep -q "ovlangvidsumm"; then
    echo "Conda environment 'ovlangvidsumm' already exists. Skipping creation."
else
    echo "Creating conda environment 'ovlangvidsumm'."
    conda create -n ovlangvidsumm python=3.10 -y
fi
conda activate ovlangvidsumm
if [ $? -ne 0 ]; then
    echo "Conda environment activation has failed. Please check."
    exit
fi
echo 'y' | conda install pip
pip install -r requirements.txt

if [ "$1" == "--skip" ]; then
    echo "Skipping OpenVINO optimized model file creation"
else
    echo "Creating OpenVINO optimized model files for MiniCPM"
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models
    python export_model.py text_generation --source_model openbmb/MiniCPM-V-2_6 --weight-format int8 --config_file_path models/config.json --model_repository_path models --target_device $DEVICE --cache 2 --pipeline_type VLM
fi
