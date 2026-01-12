FROM python:3.13-slim

ARG HTTP_PROXY_ARG
ARG HTTPS_PROXY_ARG
ARG DEBUG_ARG

ENV HTTP_PROXY=$HTTP_PROXY_ARG
ENV HTTPS_PROXY=$HTTPS_PROXY_ARG
ENV DEBUG=$DEBUG_ARG

WORKDIR /app
COPY milvus_mcp_server.py /app/
COPY milvus_wrapper.py /app/
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "milvus_mcp_server.py"]