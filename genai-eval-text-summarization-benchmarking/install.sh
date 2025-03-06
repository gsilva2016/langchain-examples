#!/bin/bash

# Install Conda
source activate-conda.sh

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping dependencies"
	activate_conda
else
	echo "Installing dependencies"
	sudo apt update
	sudo apt install wget

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
fi

# Create python enviornment
conda create -n qualbench python=3.11 -y
conda activate qualbench
pip install -r requirements.txt 
# Done
exit 0
