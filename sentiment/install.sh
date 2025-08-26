#!/bin/bash

dpkg -s sudo &> /dev/null
if [ $? != 0 ]
then
	DEBIAN_FRONTEND=noninteractive apt update
	DEBIAN_FRONTEND=noninteractive apt install sudo -y
fi

source activate-conda.sh

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping dependencies"
	activate_conda
else
	echo "Installing dependencies"
	sudo DEBIAN_FRONTEND=noninteractive apt update
	sudo DEBIAN_FRONTEND=noninteractive apt install -y curl git ffmpeg vim portaudio19-dev build-essential wget -y

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

source .env

echo "Installing sentiment analysis"
conda create -n langchain_sentiment_analysis_env python=3.12 -y --force # for a specific version
conda activate langchain_sentiment_analysis_env
echo 'y' | conda install pip
pip install -r requirements.txt --resume-retries 3


# OVMS
conda create -n $CONDA_OVMS_ENV_NAME python=3.12 -y --force
conda activate $CONDA_OVMS_ENV_NAME
echo 'y' | conda install pip
sudo DEBIAN_FRONTEND=noninteractive apt update -y && sudo DEBIAN_FRONTEND=noninteractive apt install -y libxml2 curl
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/tags/v2025.2.1/demos/common/export_models/requirements.txt -o ovms_requirements.txt
pip install -r ovms_requirements.txt

# Download OVMS
export PATH=$PATH:${PWD}/ovms/bin
if command -v ovms &> /dev/null; then
    echo "OpenVINO Model Server (OVMS) is already installed."
else
    echo "Downloading OpenVINO Model Server (OVMS)..."
    # Ubuntu 24.04
    echo "Downloading for Ubuntu 24.04..."
    wget -O ovms_ubuntu24_python_on.tar.gz https://github.com/openvinotoolkit/model_server/releases/download/v2025.2.1/ovms_ubuntu24_python_on.tar.gz
    tar -xzvf ovms_ubuntu24_python_on.tar.gz
fi

#huggingface-cli login --token $HF_ACCESS_TOKEN

curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/tags/v2025.2.1/demos/common/export_models/export_model.py -o export_model.py

rm -R models && true
mkdir models
echo "Creating OVMS models"
if [ "$device" != "GPU" ] && [ "$device" != "CPU" ]
then
	device="GPU"
fi
echo "Exporting OVMS models for inference device: $device"
python3 export_model.py text_generation --source_model $OVMS_MODEL --target_device $OVMS_DEVICE --config_file_path models/config.json --model_repository_path models --overwrite_models

