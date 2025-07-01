#!/bin/bash

set -e

# Get conversion packages
git clone https://github.com/Peterande/D-FINE.git
cd D-FINE
# Get pytorch weights
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth
# Install conversion requirements
pip install -r requirements.txt
pip install onnx onnxsim
model=l
python tools/deployment/export_onnx.py \
  --check \
  -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r dfine_l_coco.pth \
  -r dfine_l_coco.pth
python <<EOF
import openvino as ov
ov_model = ov.convert_model('dfine_l_coco.onnx')
ov.save_model(ov_model, '../ov_dfine/dfine-l-coco.xml')
EOF

# Remove extra models/scripts once done
cd ..
rm -rf D-FINE && true
