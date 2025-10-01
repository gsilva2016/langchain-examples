## Person Detection and Tracking
This module provides utilities for detecting and tracking people in video streams using DeepSORT and models from OpenVINO Model Zoo. 

### Models Used
| Task                | Model Name                              |
|---------------------|-----------------------------------------|
| Object Detection    | person-detection-0202                   |
| Re-identification   | person-reidentification-retail-0287     |

### Install
To install dependencies and download required models, follow these steps:

1. **Install Python packages:**
	 ```bash
	 pip install -q "openvino>=2024.0.0"
	 pip install -q opencv-python requests scipy tqdm "matplotlib>=3.4"
	 ```

2. **Download models:**
	 - **Person Re-identification Model**
		 ```bash
		 mkdir -p tracker_models/person-reidentification-retail-0287/FP16/
		 wget -P ./tracker_models/person-reidentification-retail-0287/FP16/ \
			 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml
		 wget -P ./tracker_models/person-reidentification-retail-0287/FP16/ \
			 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin
		 ```

	 - **Person Detection Model**
		 ```bash
		 mkdir -p tracker_models/person-detection-0202/FP16/
		 wget -P ./tracker_models/person-detection-0202/FP16/ \
			 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-0202/FP16/person-detection-0202.xml
		 wget -P ./tracker_models/person-detection-0202/FP16/ \
			 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-0202/FP16/person-detection-0202.bin
		 ```


Source code for deepsort_utils: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/person-tracking-webcam/deepsort_utils