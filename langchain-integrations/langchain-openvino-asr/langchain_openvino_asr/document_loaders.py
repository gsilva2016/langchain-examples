"""OpenVINOSpeechToTextLoader document loader."""

from typing import Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class OpenVINOSpeechToTextLoader(BaseLoader):
    """
    OpenVINOSpeechToTextLoader document loader integration

    Setup:
        Install ``langchain-openvino-asr``.

        .. code-block:: bash

            pip install -U langchain-openvino-asr

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import OpenVINOSpeechToTextLoader

            loader = OpenVINOSpeechToTextLoader(
                file_path: str = "audio.mp3",
                model_id: str = "model_id",
                device: str = "CPU",  # GPU
                return_timestamps: bool = True,
                return_language: str = "en",
                chunk_length_s: int = 30,
                load_in_8bit: bool = False,
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
        return_timestamps: bool = True,
        return_language: str = "en",
        chunk_length_s: int = 30,
        load_in_8bit: bool = False,
        batch_size: int = 1,
    ) -> None:
        """
        Initializes the OpenVINOSpeechToTextLoader.
        Args:
            file_path: A URI or local file path.
            model_id: Name of the model
            device: Hardware acclerator to utilize for inference
            return_timestamps: Enable text with corresponding timestamps for model
            return_language: Set language for model
            chunk_length_s: Length of the chunk in seconds
            load_in_8bit: Auto convert/quantize model to 8 bit if needed
            batch_size: Size of the batch
        """ # noqa: E501

        from pathlib import Path

        check_device = device.lower()
        if "gpu" != check_device and "cpu" != check_device:
            raise NotImplementedError(f"{device} not supported")

        if not Path(file_path).exists():
            raise NotImplementedError(f"{file_path} does not exist")

        self.file_path = file_path
        self.model_id = model_id
        self.device = device
        self.return_timestamps = return_timestamps
        self.return_language = return_language
        self.chunk_length_s = chunk_length_s
        self.load_in_8bit = load_in_8bit
        self.batch_size = batch_size

        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError(
                "Could not import optimum.intel.openvino python package. "
                "Please install it with `pip install optimum[openvino,nncf]`."
            ) from exc

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, load_in_8bit=self.load_in_8bit, export=False
        )

        model = model.to(self.device)
        model.compile()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            batch_size=self.batch_size,
            chunk_length_s=self.chunk_length_s,
            return_language=self.return_language,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

    def load(self) -> Iterator[Document]:
        try:
            import time

            from transformers.pipelines.audio_utils import ffmpeg_read
        except ImportError as exc:
            raise ImportError(
                "Could not import ffmpeg-python python package. "
                "Please install it with `pip install ffmpeg-python`."
            ) from exc

        audio_decoded = None
        if "gs://" in self.file_path:
            raise NotImplementedError
        elif ".mp4" in self.file_path:
            raise NotImplementedError
        elif (
            ".wav" in self.file_path
            or ".mp3" in self.file_path
            or ".m4a" in self.file_path
        ):
            with open(self.file_path, "rb") as f:
                content = f.read()
                audio_decoded = ffmpeg_read(
                    content, self.pipe.feature_extractor.sampling_rate
                )
        else:
            raise NotImplementedError("Audio file type not supported")

        audio_info = {
            "raw": audio_decoded,
            "sampling_rate": self.pipe.feature_extractor.sampling_rate,
        }

        start_time = time.time()
        chunks = self.pipe(
            audio_info,
            return_language=self.return_language,
            return_timestamps=self.return_timestamps,
        )["chunks"]

        result_total_latency = time.time() - start_time

        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "language": chunk["language"],
                    "timestamp": str(chunk["timestamp"]),
                    "result_total_latency": str(result_total_latency),
                },
            )
            for chunk in chunks
        ]

