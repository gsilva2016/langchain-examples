"""OpenVINOSpeechDiarizeLoader document loader."""

from typing import Iterator, Optional, Union, Tuple

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from concurrent.futures.thread import ThreadPoolExecutor
import requests
import os
import sys
import torch
import numpy as np


class OpenVINOSpeechDiarizeLoader(BaseLoader):
    """
    OpenVINOSpeechDiarizeLoader document loader integration

    Setup:
        Install ``langchain-openvino-diarize``.

        .. code-block:: bash

            pip install -U langchain-openvino-diarize

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import OpenVINOSpeechDiarizeLoader

            loader = OpenVINOSpeechDiarizeLoader(
                file_path: str = "audio.mp3",
                model_id: str = "model_id",
                device: str = "CPU",  # GPU
                batch_size: int = 1
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            "Transcript generated from a provided audio..."
            { "langugae": "en", "timestamp": "(0.0, 3.0)", "result_total_latency": "3" }

    """  # noqa: E501

    def __init__(
        self,
        file_path: str,
        model_id: str,
        device: str = "CPU",
        batch_size: int = 1,
        hf_token_access_token: Optional[str] = os.getenv("HF_ACCESS_TOKEN", None),
        api_base: Optional[str] = os.getenv("OPENVINO_DIARIZE_API_BASE", None)
    ) -> None:
        """
        Initializes the OpenVINOSpeechDiarizeLoader.
        Args:
            file_path: A URI or local file path.
            model_id: Name of the model
            device: Hardware acclerator to utilize for inference
            batch_size: Size of the batch
        """ # noqa: E501
        if hf_token_access_token is None:
            print("export HF_ACCESS_TOKEN=<YOUR_ACCESS_TOKEN> is necessary to download the model from HuggingFace.")
            print("Please refer to https://huggingface.co/pyannote/speaker-diarization-3.1 for accepting model terms.")
            print("For more information on user access tokens for access to gated models see https://huggingface.co/docs/hub/en/security-tokens")
            sys.exit(1)

        self.api_base = api_base
        self.file_path = file_path

        if not api_base is None:
            return

        self.device = device
        self.model_id = model_id
        self.batch_size = batch_size

        from pathlib import Path

        device = device.lower()
        if "gpu" != device and "cpu" != device:
            raise NotImplementedError(f"{device} not supported")

        if "gpu" == device:
            device = "xpu"

        if not Path(file_path).exists():
            raise NotImplementedError(f"{file_path} does not exist")

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "Could not import pyannote.audio python package. "
            ) from exc

        pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=hf_token_access_token)
        pipeline.to(torch.device(device))
        self.pipeline = pipeline

    def post_request(self, input_data: str):
        request = { 'file': open(input_data, 'rb')}
        response = requests.post(url=self.api_base, files=request)
        return response.content


    def load(self) -> Iterator[Document]:
        if not self.api_base is None:
            # send the request to the FastAPI endpoint using a ThreadPoolExecutor for async processing
            with ThreadPoolExecutor() as pool:
                # toto: place holder for including diarize results
                future = pool.submit(self.post_request, self.file_path)
                future_res = future.result().decode("utf-8")
            return future_res

        import time

        if "gs://" in self.file_path:
            raise NotImplementedError
        elif ".mp4" in self.file_path:
            raise NotImplementedError
        elif (
            ".wav" in self.file_path
            or ".mp3" in self.file_path
            or ".m4a" in self.file_path
        ):
            pass
        else:
            raise NotImplementedError("Audio file type not supported")

        AUDIO_FILE = {'uri': self.file_path.split('.')[0], 'audio': self.file_path}
        start_time = time.time()
        diarization = self.pipeline(AUDIO_FILE)

        docs = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            doc = Document(
                    page_content=f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}",
                    metadata={
                        "speaker": speaker,
                        "start": f"{turn.start:.1f}",
                        "stop": f"{turn.end:.1f}",
                    }
            )

            docs.append(doc)

        result_total_latency = time.time() - start_time
        return docs
