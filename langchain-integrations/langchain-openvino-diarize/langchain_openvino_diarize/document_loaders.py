"""OpenVINOSpeechDiarizeLoader document loader."""

from typing import Iterator, Optional, Union, Tuple

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from concurrent.futures.thread import ThreadPoolExecutor
import requests
import os
import sys
import torch
#import openvino.torch
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
        Initializes the OpenVINOSpeechToTextLoader.
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

        check_device = device.lower()
        if "gpu" != check_device and "cpu" != check_device:
            raise NotImplementedError(f"{device} not supported")

        if not Path(file_path).exists():
            raise NotImplementedError(f"{file_path} does not exist")

        try:
            from pyannote.audio import Pipeline
            #from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            #from transformers import AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError(
                "Could not import pyannote.audio python package. "
                "Please install it with `pip install git+https://github.com/eaidova/pyannote-audio.git@hub0.10`."
            ) from exc

        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError(
                "Could not import openvino python package. "
                "Please install it with pip install openvino"
            ) from exc

        from pathlib import Path
        pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=hf_token_access_token)
        pipeline.segmentation.model = pipeline.segmentation.model.to('xpu')

        #core = ov.Core()
        #ov_speaker_segmentation_path = Path("pyannote-segmentation.xml")

        #if not ov_speaker_segmentation_path.exists():
            #onnx_path = ov_speaker_segmentation_path.with_suffix(".onnx")
        #    print("-------->Batch Size for export:", self.batch_size)
            #torch.onnx.export(
            #    pipeline._segmentation.model, 
            #    torch.zeros((1, 1, 160000)), 
            #    onnx_path, 
            #    input_names=["chunks"], 
            #    output_names=["outputs"], 
            #    opset_version = 11,
            #    do_constant_folding=False,
#                dynamic_axes={"chunks": {0: "batch_size", self.batch_size: "wave_len"}}
            #)
            # remove dynamic rank by specifying input shape
#            ov_speaker_segmentation = ov.convert_model(onnx_path, input=[1,1,160000])
            #ov_speaker_segmentation = ov.convert_model(onnx_path, example_input=torch.rand(1,1,160000), input=[1,1,160000])
#            ov_speaker_segmentation = ov.convert_model(pipeline._segmentation.model)
            #ov.save_model(ov_speaker_segmentation, str(ov_speaker_segmentation_path))
        #    print(f"Diarization model successfully cached to {ov_speaker_segmentation_path}")
        #else:
        #    ov_speaker_segmentation = core.read_model(ov_speaker_segmentation_path)
        #    print(f"Diarization model successfully loaded from {ov_speaker_segmentation_path}")

        #pipeline._segmentation.model = pipeline._segmentation.model.to_torchscript(method="trace")
        #pipeline._segmentation.model = torch.compile(pipeline._segmentation.model, backend='openvino', options={"device" : "CPU" , "config" : {"PERFORMANCE_HINT" : "LATENCY"}}) # core.compile_model(ov_speaker_segmentation, self.device)
        #pipeline._segmentation.model = ov.convert_model(pipeline._segmentation.model, example_input=torch.rand(1,1,160000))
#        infer_request = self.ov_seg_model.create_infer_request()
#        self.ov_seg_out = self.ov_seg_model.output(0)

        self.pipeline = pipeline
        #self.pipeline._segmentation.infer = self.infer_segm

    def infer_segm(self, chunks: torch.Tensor) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """
        Inference speaker segmentation mode using OpenVINO
        Parameters:
            chunks (torch.Tensor) input audio chunks
        Return:
            segments (np.ndarray)
        """

        res = self.ov_seg_model(chunks)

        ret = res[self.ov_seg_out]        
        print("Sending ret: ", ret.shape)
        print("ret: ", ret)
        print("res: ", res)
        print("res shape: ", res.shape)
        return ret

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
            #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
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
        print("Diarization Latency: ", result_total_latency)
        return docs
