#!/bin/bash

source .env

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN environment variable in .env file"
    exit 1
fi

dpkg -s sudo &> /dev/null
if [ $? != 0 ]
then
	DEBIAN_FRONTEND=noninteractive apt update
	DEBIAN_FRONTEND=noninteractive apt install sudo -y
fi

# check if curl is installed
if ! command -v curl &> /dev/null; then
    echo "curl is not installed. Installing curl"
    sudo DEBIAN_FRONTEND=noninteractive apt install curl -y
fi

# Installing Milvus and Docker
echo "Installing Milvus as a standalone service"
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing Docker"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh

    if [ $? -ne 0 ]; then
        echo "Docker installation failed. Please check the logs."
        exit 1
    fi

    # Add user to the docker group to prevent permission issues
    sudo groupadd docker
    sudo usermod -aG docker $USER

    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "Docker has been installed. Now re-run ./install.sh to apply the Docker group changes. Else container will not load due to permission issues."
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    newgrp docker
fi

echo "Docker is installed. Proceeding with Milvus setup"
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

    if [ $? -ne 0 ]; then
        echo "Milvus failed to start. Please check the logs."
        exit 1
    fi

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

# Install OpenVINO Model Server (OVMS) on baremetal
if [ "$1" == "--skip" ]; then
    bash install-ovms.sh --skip
else
    bash install-ovms.sh
fi

if [ $? -ne 0 ]; then
    echo "OpenVINO Model Server (OVMS) installation failed. Please check the logs."
    exit 1
fi

echo "OpenVINO Model Server (OVMS) installation and creation of optimized model files completed successfully."
echo ""

# Create python environment
echo "Creating conda environment $CONDA_ENV_NAME."
conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME
if [ $? -ne 0 ]; then
    echo "Conda environment activation has failed. Please check."
    exit
fi

echo 'y' | conda install pip
pip install -r requirements.txt

echo "All installation steps completed successfully."


