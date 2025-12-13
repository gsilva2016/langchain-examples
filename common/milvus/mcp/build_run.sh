#!/bin/bash

source .env

if docker ps | grep -q milvus; then
    echo "Milvus is already running."
    echo ""

else
    echo "Start Milvus first. Refer to README.md for setup instructions."
    exit 1
fi

MILVUS_DIR=$(realpath ../)

cp $MILVUS_DIR/milvus_wrapper.py milvus_wrapper.py
docker compose up --build -d

rm milvus_wrapper.py