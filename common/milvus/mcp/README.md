# Milvus MCP Server

This MCP server exposes Milvus vector database operations as MCP tools. It uses FastMCP for creating the server and runs as a HTTP server on a configurable port 8001. It allows you to manage Milvus collections, insert and search vectors, and perform other database operations via a MCP client or other agentic workflows.

## MCP Tools Exposed
1. `list_collections()`: List all collections

2. `create_collection(collection_name, dimension, overwrite=False)`: Create a collection

3. `insert_data(collection_name, vectors, metadatas, partition_name=None)`: Insert vectors

4. `search(collection_name, query_vector, output_fields=["metadata", "vector"], limit=1, filter="", partition_names=None)`: Search for similar vectors

5. `query(collection_name, filter="", limit=100, output_fields=["pk", "metadata"], partition_names=None)`: Query data

6. `upsert_data(collection_name, pks, vectors, metadatas, partition_name=None)`: Upsert data

7. `drop_collection(collection_name)`: Delete a collection

## Usage

### Environment Variables

Set required env variables in `milvus/mcp/.env`

### Ensure Milvus is up and running

1. If you ran `video-summarization/install.sh`, Milvus will be installed and up/running. Verify via `docker ps`

2. Else, run these commands:

```
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
```

### Build and Run with Docker

```
./build_run.sh
```

### Run Locally without Docker

```
conda activate ovlangvidsumm (reuse from your video-summarization env, else conda create -n test python=3.10 -y)
pip install -r requirements.txt
PYTHONPATH=.. python milvus_mcp_server.py
```

## Example Client Usage

See `test_mcp_client.py` for an example of how to connect to the MCP server and call tools.
