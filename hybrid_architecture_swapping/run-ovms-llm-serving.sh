#!/bin/bash

sudo docker run -itd --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) ovms-llm-serving:1.0 --rest_port 8000 --config_path models/config_all.json

