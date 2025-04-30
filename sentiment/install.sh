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


echo "Installing sentiment analysis"
conda create -n langchain_sentiment_analysis_env python=3.10.12 -y --force # for a specific version
conda activate langchain_sentiment_analysis_env
echo 'y' | conda install pip
# need for export. current reqs.txt in ovms repo does not have all reqs needed
pip install -r requirements.txt --resume-retries 3

rm -R model_server && true
mkdir model_server && cd model_server
# latest 2025.1 does not have LLM support so pulling files last commit instead
#git clone -b v2025.1 --single-branch https://github.com/openvinotoolkit/model_server.git # doesnt include quantize for LLMs
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/04e4909c11cf394e3bc41784b0e00f6506ba843b/demos/common/export_models/requirements.txt -O requirements.txt
pip install -r requirements.txt --resume-retries 3
cd ..
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/04e4909c11cf394e3bc41784b0e00f6506ba843b/demos/common/export_models/export_model.py -O export_model.py

rm -Rf ovms && true
wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu22_python_on.tar.gz -O ovms_ubuntu22_python_on.tar.gz
tar -xzvf ovms_ubuntu22_python_on.tar.gz

rm -R ovms_models && true
mkdir ovms_models
echo "Creating OVMS models"
if [ "$device" != "GPU" ] && [ "$device" != "CPU" ]
then
	device="GPU"
fi
echo "Exporting OVMS models for inference device: $device"
python3 export_model.py text_generation --source_model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --target_device $device --config_file_path ovms_models/config_all.json --model_repository_path ovms_models --overwrite_models
