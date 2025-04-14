# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6

## Installation

1. First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain
access to the model. For more information on user access tokens for access to gated models
see [here](https://huggingface.co/docs/hub/en/security-tokens).
2. Next, install Intel Client GPU, Conda, Set Up Python Environment and Create OpenVINO optimized model for MiniCPM. Ensure you `export HUGGINGFACE_TOKEN=<MY_TOKEN_HERE>` before executing the below command.

```
# Validated on Ubuntu 24.04 and 22.04
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only, the following
command can be used to skip the re-install of dependencies. 

```
./install.sh --skip
```

## Convert and Save Optimized MiniCPM-V-2_6

This section can be skipped if you ran `install.sh` the first time. The `install.sh` script runs this command as part of 
its setup. This section is to give the user flexibility to tweak the `optimum-cli` command for certain model parameters. 

Ensure you `export HUGGINGFACE_TOKEN=<MY_TOKEN_HERE>` before executing the below command. 
```
source activate-conda.sh
activate_conda
conda activate ovlangvidsumm
huggingface-cli login --token $HUGGINGFACE_TOKEN
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int8 MiniCPM_INT8 # int4 also available
```

## Run Video Summarization

Summarize [this sample video](https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4)
using `video_summarizer.py`.

```
./run-demo.sh 
```

Note: if the demo has already been run, you can use the following command to skip the video download.

```
./run-demo.sh --skip
```
