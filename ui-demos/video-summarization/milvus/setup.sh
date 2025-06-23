#!/bin/bash

if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing Docker"

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
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

echo "Docker is installed. Proceeding with Milvus setup..."

echo "Downloading and running Milvus"
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start
echo "Milvus has been started. You can access it at http://localhost:19530"

echo "You can stop Milvus using the following command:"
echo "bash standalone_embed.sh stop"
echo "You can delete Milvus data using the following command:"
echo "bash standalone_embed.sh delete"
